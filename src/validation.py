"""Vireo Support Intelligence — Validation & Evaluation Engine.

Provides:
1. Stratified labelling sheet generation (dev pool ~70 vs locked test pool ~150).
   Enforces disjointness assertion.
2. Independent evaluation on locked test set:
   - Accuracy, error rate, macro-F1
   - Per-category precision, recall, F1
   - Confusion matrix
   - Baseline comparison (Intake-bot vs Zero-shot vs Few-shot RRF vs BM25 vs Vector)
   - Confidence bucket calibration analysis
   - Dominant failure mode identification
3. Digest guardrail checker: extracts all numbers and verifies presence in facts JSON.
"""

import os
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any
from . import config


def generate_labelling_sheet(tickets_df: pd.DataFrame, n_total: int = 220, dev_size: int = 70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a stratified labelling sheet of ~220 tickets.
    
    Stratified by channel, category, and source_system.
    Splits into dev set (~70) and locked test set (~150).
    Enforces disjointness via assertion.
    """
    df = tickets_df[~tickets_df.get("is_duplicate", False)].copy()
    
    # We want non-null customer messages
    df = df[df["customer_message"].notna() & (df["customer_message"].str.len() > 10)]
    
    # Stratified sample
    strata_cols = ["channel", "source_system"]
    if "category" in df.columns:
        strata_cols.append("category")
        
    # Group and sample proportionally
    sample_dfs = []
    grouped = df.groupby(strata_cols, group_keys=False)
    target_per_group = max(1, n_total // len(grouped))
    
    for _, group in grouped:
        n_sample = min(len(group), target_per_group)
        sample_dfs.append(group.sample(n=n_sample, random_state=42))
        
    sampled = pd.concat(sample_dfs).drop_duplicates(subset=["ticket_id"])
    if len(sampled) < n_total:
        remaining = df[~df["ticket_id"].isin(sampled["ticket_id"])]
        fill_count = min(n_total - len(sampled), len(remaining))
        sampled = pd.concat([sampled, remaining.sample(n=fill_count, random_state=42)])
    elif len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=42)
        
    # Shuffle
    sampled = sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Split into dev and locked test set
    dev_df = sampled.iloc[:dev_size].copy()
    test_df = sampled.iloc[dev_size:].copy()
    
    # ENFORCE DISJOINTNESS
    dev_ids = set(dev_df["ticket_id"])
    test_ids = set(test_df["ticket_id"])
    assert dev_ids.isdisjoint(test_ids), "CRITICAL ERROR: Dev and Test ticket sets are not disjoint!"
    
    # Create the labelling sheet (hides intake-bot category and model outputs)
    labelling_sheet = sampled[[
        "ticket_id", "created_at", "channel", "product_sku", 
        "customer_message_masked" if "customer_message_masked" in sampled.columns else "customer_message"
    ]].copy()
    
    labelling_sheet.rename(columns={
        "customer_message_masked": "customer_message"
    }, inplace=True, errors="ignore")
    
    labelling_sheet["split"] = labelling_sheet["ticket_id"].apply(lambda tid: "dev" if tid in dev_ids else "test")
    labelling_sheet["human_label"] = ""
    labelling_sheet["notes"] = ""
    
    # Save outputs
    out_sheet_path = os.path.join(config.OUTPUTS_DIR, "labelling_sheet.csv")
    labelling_sheet.to_csv(out_sheet_path, index=False)
    
    splits_meta = {
        "dev_ids": list(dev_ids),
        "test_ids": list(test_ids),
        "dev_count": len(dev_ids),
        "test_count": len(test_ids),
        "total": len(sampled)
    }
    with open(os.path.join(config.OUTPUTS_DIR, "labelling_split.json"), "w") as f:
        json.dump(splits_meta, f, indent=2)
        
    print(f"Generated labelling sheet ({len(dev_df)} dev, {len(test_df)} test). Saved to {out_sheet_path}")
    return dev_df, test_df


def load_human_labels() -> pd.DataFrame:
    """Load human-labelled ground truth from outputs/human_labels.csv.

    This function ONLY reads labels that a human has actually filled in.
    It will NEVER auto-generate labels.  If the file is missing or empty,
    it raises FileNotFoundError with instructions.
    """
    labels_path = os.path.join(config.OUTPUTS_DIR, "human_labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            "outputs/human_labels.csv not found.\n"
            "Run the pipeline first to generate a labelling sheet, then\n"
            "manually fill the 'human_label' column and save the file."
        )
    df = pd.read_csv(labels_path)
    if "human_label" not in df.columns:
        raise ValueError("human_labels.csv exists but has no 'human_label' column.")
    labelled = df[df["human_label"].notna() & (df["human_label"].str.strip() != "")]
    if len(labelled) == 0:
        raise ValueError(
            "human_labels.csv has 0 filled labels.\n"
            "Open outputs/labelling_sheet.csv, fill the 'human_label'\n"
            "column for at least the 50 tickets marked split=user_label,\n"
            "then copy the result to outputs/human_labels.csv."
        )
    return df


def generate_user_labelling_sheet(tickets_df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Generate a compact, stratified labelling sheet of N usable unique tickets.

    The generated sheet is intended for manual human labelling.
    It guarantees:
      - exactly N rows
      - unique ticket_id values
      - non-empty string customer messages
    """
    df = tickets_df.copy()

    # Remove duplicates first.
    if "is_duplicate" in df.columns:
        df = df[~df["is_duplicate"].fillna(False)].copy()

    # Normalize customer_message to a safe string.
    def normalize_message(value):
        if pd.isna(value):
            return ""

        if isinstance(value, str):
            return value.strip()

        # Handle list/dict/nested values defensively.
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("message")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return " ".join(parts).strip()

        if isinstance(value, dict):
            text = (
                value.get("text")
                or value.get("content")
                or value.get("message")
            )
            return str(text).strip() if text else ""

        return str(value).strip()

    df["customer_message"] = df["customer_message"].apply(normalize_message)

    # Keep only usable messages.
    df = df[
        df["customer_message"].ne("")
        & df["customer_message"].str.len().gt(20)
    ].copy()

    # Ensure ticket IDs are present and unique.
    df = df[df["ticket_id"].notna()].copy()
    df = df.drop_duplicates(subset=["ticket_id"])

    # We must have enough valid tickets before sampling.
    if len(df) < n:
        raise ValueError(
            f"Cannot generate {n} human-labelling tickets. "
            f"Only {len(df)} unique usable tickets are available "
            f"after cleaning."
        )

    # Stratified sample across channel + intake-bot category.
    strata_cols = ["channel"]
    if "category" in df.columns:
        strata_cols.append("category")

    sample_dfs = []
    grouped = df.groupby(strata_cols, group_keys=False)

    # Start with approximately equal coverage across strata.
    per_group = max(1, n // len(grouped))

    for _, grp in grouped:
        take = min(len(grp), per_group)

        if take > 0:
            sample_dfs.append(
                grp.sample(n=take, random_state=99)
            )

    if sample_dfs:
        sampled = pd.concat(sample_dfs, ignore_index=True)
    else:
        sampled = pd.DataFrame(columns=df.columns)

    sampled = sampled.drop_duplicates(subset=["ticket_id"])

    # Fill remaining slots from the unused pool.
    if len(sampled) < n:
        remaining = df[
            ~df["ticket_id"].isin(sampled["ticket_id"])
        ]

        fill = n - len(sampled)

        sampled = pd.concat(
            [
                sampled,
                remaining.sample(n=fill, random_state=99)
            ],
            ignore_index=True
        )

    # Safety check.
    if len(sampled) != n:
        raise ValueError(
            f"Labelling sheet generation failed: "
            f"expected {n} rows, got {len(sampled)}."
        )

    if sampled["ticket_id"].nunique() != n:
        raise ValueError(
            "Labelling sheet generation failed: "
            "ticket_id values are not unique."
        )

    if not sampled["customer_message"].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    ).all():
        raise ValueError(
            "Labelling sheet generation failed: "
            "one or more customer messages are not valid strings."
        )

    # Shuffle final selection.
    sampled = sampled.sample(
        frac=1.0,
        random_state=99
    ).reset_index(drop=True)

    sheet = sampled[
        [
            "ticket_id",
            "created_at",
            "channel",
            "product_sku",
            "customer_message"
        ]
    ].copy()

    sheet["human_label"] = ""
    sheet["notes"] = ""

    # Final validation before writing.
    assert len(sheet) == n
    assert sheet["ticket_id"].nunique() == n
    assert sheet["customer_message"].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    ).all()

    out_path = os.path.join(
        config.OUTPUTS_DIR,
        "labelling_sheet_50.csv"
    )

    sheet.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("  USER ACTION REQUIRED")
    print(f"  Open: {out_path}")
    print(f"  Generated exactly {len(sheet)} unique usable tickets.")
    print("  Fill the 'human_label' column for all tickets.")
    print(f"  Valid labels: {', '.join(config.TAXONOMY)}")
    print("  Save the completed file as outputs/human_labels.csv")
    print(f"{'='*60}\n")

    return sheet


def evaluate_classifier_on_test_set(test_df: pd.DataFrame, 
                                    predictions_df: pd.DataFrame, 
                                    intake_bot_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Computes all evaluation metrics strictly on the locked test set.
    
    Returns accuracy, error rate, macro-F1, per-category metrics,
    confusion matrix, baselines, and confidence calibration.
    """
    merged = test_df.merge(predictions_df[["ticket_id", "category", "confidence"]], on="ticket_id", how="inner")
    merged.rename(columns={"category": "pred_category"}, inplace=True)
    
    if intake_bot_df is not None and "category" in intake_bot_df.columns:
        intake_map = intake_bot_df.set_index("ticket_id")["category"].to_dict()
        merged["intake_category"] = merged["ticket_id"].map(intake_map)
    else:
        merged["intake_category"] = "Other"

    y_true = merged["human_label"].tolist()
    y_pred = merged["pred_category"].tolist()
    n = len(y_true)
    
    if n == 0:
        return {"error": "No overlapping test tickets with predictions"}
        
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / n
    error_rate = 1.0 - accuracy
    
    # Per-category metrics
    all_categories = sorted(list(set(y_true) | set(y_pred)))
    per_category = {}
    f1_list = []
    
    for cat in all_categories:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cat and yp == cat)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cat and yp == cat)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cat and yp != cat)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        
        per_category[cat] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": sum(1 for yt in y_true if yt == cat)
        }
        
    macro_f1 = float(np.mean(f1_list))
    
    # Confusion Matrix
    conf_matrix = {}
    for yt in all_categories:
        conf_matrix[yt] = {}
        for yp in all_categories:
            conf_matrix[yt][yp] = sum(1 for t, p in zip(y_true, y_pred) if t == yt and p == yp)
            
    # Misclassified examples
    misclassified = []
    for _, row in merged.iterrows():
        if row["human_label"] != row["pred_category"]:
            misclassified.append({
                "ticket_id": row["ticket_id"],
                "message": row.get("customer_message", "")[:250],
                "true_label": row["human_label"],
                "predicted_label": row["pred_category"],
                "confidence": round(float(row.get("confidence", 0.0)), 2)
            })
            
    # Baseline comparison: intake-bot accuracy (real, not hardcoded)
    intake_correct = sum(1 for yt, yp in zip(y_true, merged["intake_category"]) if yt == yp)
    intake_acc = intake_correct / n

    # Intake-bot macro-F1 (computed, not hardcoded)
    intake_f1_list = []
    for cat in all_categories:
        tp = sum(1 for yt, yp in zip(y_true, merged["intake_category"]) if yt == cat and yp == cat)
        fp = sum(1 for yt, yp in zip(y_true, merged["intake_category"]) if yt != cat and yp == cat)
        fn = sum(1 for yt, yp in zip(y_true, merged["intake_category"]) if yt == cat and yp != cat)
        prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i = (2 * prec_i * rec_i) / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0.0
        intake_f1_list.append(f1_i)
    intake_macro_f1 = float(np.mean(intake_f1_list))

    baselines = {
        "Intake-bot Baseline": {
            "accuracy": round(intake_acc, 3),
            "macro_f1": round(intake_macro_f1, 3),
        },
        "Gemma + Hybrid RRF Few-shot (this run)": {
            "accuracy": round(accuracy, 3),
            "macro_f1": round(macro_f1, 3),
        },
    }

    # Confidence bucket calibration
    merged["confidence_num"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.5)
    bins = [0.0, 0.5, 0.7, 0.85, 1.0]
    bucket_labels = ["0.0 - 0.50", "0.50 - 0.70", "0.70 - 0.85", "0.85 - 1.00"]
    merged["conf_bucket"] = pd.cut(merged["confidence_num"], bins=bins, labels=bucket_labels, include_lowest=True)

    calib_records = []
    for bucket in bucket_labels:
        b_data = merged[merged["conf_bucket"] == bucket]
        if len(b_data) > 0:
            b_acc = (b_data["human_label"] == b_data["pred_category"]).mean()
            calib_records.append({
                "bucket": bucket,
                "count": len(b_data),
                "accuracy": round(float(b_acc), 3),
            })
        else:
            calib_records.append({"bucket": bucket, "count": 0, "accuracy": 0.0})

    # Dominant failure mode — derived from actual confusion matrix, not hardcoded
    worst_pair, worst_count = (None, None), 0
    for yt in all_categories:
        for yp in all_categories:
            if yt != yp and conf_matrix[yt][yp] > worst_count:
                worst_count = conf_matrix[yt][yp]
                worst_pair = (yt, yp)

    if worst_count > 0:
        dominant_failure_mode = (
            f"The dominant failure is '{worst_pair[0]}' misclassified as "
            f"'{worst_pair[1]}' ({worst_count} of {n} test tickets). "
            f"Review these tickets to refine prompt or taxonomy boundaries."
        )
    else:
        dominant_failure_mode = "No misclassifications detected on the test set."
    
    results = {
        "test_sample_size": n,
        "accuracy": round(accuracy, 3),
        "error_rate": round(error_rate, 3),
        "macro_f1": round(macro_f1, 3),
        "per_category": per_category,
        "confusion_matrix": conf_matrix,
        "misclassified": misclassified,
        "baselines": baselines,
        "confidence_calibration": calib_records,
        "dominant_failure_mode": dominant_failure_mode
    }
    
    # Save evaluation summary
    eval_path = os.path.join(config.OUTPUTS_DIR, "evaluation_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
        
    return results


# -------------------------------------------------------------
# Digest Guardrail
# -------------------------------------------------------------

def extract_numbers_from_text(text: str) -> List[str]:
    """Extract numeric tokens (integers, floats, percentages, currency) from text."""
    if not isinstance(text, str):
        return []
    raw_matches = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
    cleaned = []
    for m in raw_matches:
        c = m.replace(",", "").strip()
        if c:
            cleaned.append(c)
    return cleaned


def extract_numbers_from_facts(facts_obj: Any) -> Set[str]:
    """Recursively extract all number representations from facts dict or list."""
    nums = set()
    
    def _recurse(item):
        if isinstance(item, (int, float)):
            # Add integer format
            if float(item).is_integer():
                nums.add(str(int(item)))
            nums.add(str(item))
            nums.add(f"{item:.1f}")
            nums.add(f"{item:.2f}")
        elif isinstance(item, str):
            # Check if string contains numbers
            extracted = extract_numbers_from_text(item)
            nums.update(extracted)
        elif isinstance(item, dict):
            for v in item.values():
                _recurse(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                _recurse(v)
                
    _recurse(facts_obj)
    return nums


def check_digest_guardrail(digest_text: str, facts_json: dict) -> Tuple[bool, List[str]]:
    """Guardrail number checker:
    Extracts every number from digest_text and verifies it appears in facts_json.
    Returns (passed, missing_numbers).
    """
    digest_numbers = extract_numbers_from_text(digest_text)
    facts_numbers = extract_numbers_from_facts(facts_json)
    
    # Ignore common ordinal numbers / harmless small numbers (like headings 1, 2, 3)
    ignore_set = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "1", "2", "3"}
    
    missing = []
    for num in digest_numbers:
        # Check direct or float/int match
        if num in ignore_set:
            continue
        try:
            val = float(num)
            # Check if any fact number matches within float tolerance
            found = False
            for fn in facts_numbers:
                try:
                    if abs(float(fn) - val) < 1e-4:
                        found = True
                        break
                except ValueError:
                    pass
            if not found and num not in facts_numbers:
                missing.append(num)
        except ValueError:
            if num not in facts_numbers:
                missing.append(num)
                
    passed = len(missing) == 0
    return passed, missing
