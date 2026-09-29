"""Vireo Support Intelligence — End-to-End Pipeline Runner.

Usage:
    python -m src.pipeline [--limit N] [--skip-embed] [--skip-classify]

Executes:
1. Data cleaning & enrichment (preprocessing.py)
2. Hybrid search indexing & embeddings (retrieval.py)
3. Ticket classification with caching & resume (classifier.py)
4. Repeat-contact detection & colleague claim analysis (repeat_contacts.py)
5. Weekly metrics, breach analysis & agent leaderboard (analytics.py)
6. Business opportunity evaluation (opportunity.py)
7. Validation sheet & test evaluation (validation.py)
8. Weekly executive digest generation with guardrails (digest.py)
"""

import argparse
import os
import sys
import json
import pandas as pd
from . import config
from .data_loader import load_agents
from .preprocessing import preprocess
from .retrieval import HybridRetriever
from .classifier import classify_all
from .repeat_contacts import compute_repeat_contacts, analyse_colleague_claim
from .analytics import compute_weekly_metrics, compute_agent_metrics, analyse_breaches_by_hour
from .opportunity import compute_opportunities
from .validation import (
    generate_labelling_sheet,
    load_human_labels,
    generate_user_labelling_sheet,
    evaluate_classifier_on_test_set,
)
from .digest import build_weekly_facts, generate_digest


def run_pipeline(limit: int = None, skip_embed: bool = False, skip_classify: bool = False):
    """Run full reproducible pipeline."""
    print("=" * 65)
    print("  VIREO SUPPORT INTELLIGENCE — PIPELINE RUNNER")
    print(f"  Limit: {limit or 'Full dataset'}")
    print(f"  Ollama Chat Model: {config.OLLAMA_MODEL}")
    print(f"  Ollama Embed Model: {config.OLLAMA_EMBED_MODEL}")
    print("=" * 65)
    
    # ---------------------------------------------------------
    # STEP 1: Preprocessing
    # ---------------------------------------------------------
    print("\n[1/8] Running Preprocessing & Data Cleaning...")
    tickets_df = preprocess(limit=limit)
    print(f"  Processed {len(tickets_df)} tickets ({tickets_df['is_duplicate'].sum()} duplicates flagged)")
    
    # Also save parquet if available
    try:
        parquet_path = os.path.join(config.OUTPUTS_DIR, "tickets_clean.parquet")

        import pyarrow as pa
        import pyarrow.parquet as pq

        arrow_table = pa.Table.from_pandas(
            tickets_df,
            preserve_index=False
        )

        pq.write_table(arrow_table, parquet_path)

        print(f"  Saved Parquet to {parquet_path}")

    except Exception as e:
        print(f"  Parquet export skipped: {type(e).__name__}: {e}")

    # ---------------------------------------------------------
    # STEP 2: Validation Split & Labelling Setup
    # ---------------------------------------------------------
    print("\n[2/8] Setting up Human Validation & Labelling Sheet...")
    dev_df, test_df = generate_labelling_sheet(tickets_df)
    dev_ids = set(dev_df["ticket_id"])
    test_ids = set(test_df["ticket_id"])
    print(f"  Validation setup: {len(dev_ids)} dev examples, {len(test_ids)} locked test examples")

    # Generate 50-ticket labelling sheet for user if it doesn't exist
    label_sheet_path = os.path.join(config.OUTPUTS_DIR, "labelling_sheet_50.csv")
    if not os.path.exists(label_sheet_path):
        generate_user_labelling_sheet(tickets_df, n=50)
    else:
        print(f"  User labelling sheet already exists at {label_sheet_path}")

    # Try loading human labels (will fail gracefully if user hasn't labelled yet)
    human_labels_df = None
    try:
        human_labels_df = load_human_labels()
        print(f"  Loaded {len(human_labels_df)} human labels")
    except (FileNotFoundError, ValueError) as e:
        print(f"  ⚠ Human labels not available: {e}")
        print(f"  → Evaluation step will be skipped. Label the 50-ticket sheet and re-run.")

    # ---------------------------------------------------------
    # STEP 3: Hybrid Retrieval Index
    # ---------------------------------------------------------
    print("\n[3/8] Building Hybrid Retrieval Index (BM25 + Vectors)...")
    valid_texts = tickets_df["customer_message_masked"].fillna("").tolist()
    valid_ids = tickets_df["ticket_id"].tolist()
    retriever = HybridRetriever(texts=valid_texts, ids=valid_ids)
    
    if not skip_embed:
        try:
            # Embed ALL texts upfront using batch API (Ollama /api/embed
            # accepts a list). This prevents per-ticket API calls during
            # classification which were the main speed bottleneck.
            print(f"  Pre-indexing {len(valid_texts)} document embeddings (batch API)...")
            retriever.embed_texts(valid_texts, batch_size=50)
            print(f"  Embedding index built successfully")
        except Exception as e:
            print(f"  Embedding index note: {e}")

    # ---------------------------------------------------------
    # STEP 4: Classification
    # ---------------------------------------------------------
    print("\n[4/8] Running Taxonomy Classification & Cache...")
    cache_path = os.path.join(config.OUTPUTS_DIR, "classified_tickets.csv")
    
    # If skip_classify or running smoke test with pre-existing cache
    if skip_classify:
        if os.path.exists(cache_path):
            print(f"  Using existing classification cache at {cache_path}")
            classified_df = pd.read_csv(cache_path, dtype=str)
        else:
            print("  Initializing precomputed classification cache...")
            classified_df = _create_fallback_classification_cache(tickets_df)
    else:
        # If limit is specified for smoke test, classify only the limited sample
        classify_subset = tickets_df.head(limit) if limit else tickets_df
        try:
            classified_df = classify_all(
                classify_subset,
                retriever=retriever,
                prompt_version="v1",
                batch_save_every=10,
                dev_ids=dev_ids,
                test_ids=test_ids
            )
        except Exception as e:
            print(f"  Classification error ({e}), building rule-based cache fallback...")
            classified_df = _create_fallback_classification_cache(tickets_df)

    # Merge classified category back to tickets_df
    if "category" in classified_df.columns:
        cat_map = classified_df.set_index("ticket_id")["category"].to_dict()
        tickets_df["classified_category"] = tickets_df["ticket_id"].map(cat_map).fillna(tickets_df["category"])
    else:
        tickets_df["classified_category"] = tickets_df["category"]
        
    # Re-save tickets_clean with classified_category
    tickets_df.to_csv(os.path.join(config.OUTPUTS_DIR, "tickets_clean.csv"), index=False)

    # ---------------------------------------------------------
    # STEP 5: Repeat Contact Detection
    # ---------------------------------------------------------
    print("\n[5/8] Computing Repeat Contacts & Colleague Claim Analysis...")
    pairs_df = compute_repeat_contacts(tickets_df, strict_threshold=5, loose_threshold=3)
    colleague_claim = analyse_colleague_claim(tickets_df, pairs_df)
    with open(os.path.join(config.OUTPUTS_DIR, "colleague_claim_analysis.json"), "w") as f:
        json.dump(colleague_claim, f, indent=2)
    print("  Colleague claim test complete. Results saved to outputs/colleague_claim_analysis.json")

    # ---------------------------------------------------------
    # STEP 6: Weekly Analytics, Leaderboard & Breach Profiling
    # ---------------------------------------------------------
    print("\n[6/8] Generating Weekly Analytics, Agent Leaderboard & Breach Profile...")
    weekly_df = compute_weekly_metrics(tickets_df)
    agents_df = load_agents()
    agent_metrics = compute_agent_metrics(tickets_df, agents_df)
    breach_analysis = analyse_breaches_by_hour(tickets_df)
    print(f"  Weekly metrics computed for {len(weekly_df)} weeks")

    # ---------------------------------------------------------
    # STEP 7: Business Opportunity Engine
    # ---------------------------------------------------------
    print("\n[7/8] Evaluating Business Opportunities (Observed / Estimated / Scenario)...")
    opps_df = compute_opportunities(tickets_df, weekly_df, pairs_df)
    top_opp = opps_df.iloc[0] if len(opps_df) > 0 else None
    if top_opp is not None:
        print(f"  WINNER OPPORTUNITY: {top_opp['opportunity']} (Saving ~Rs {top_opp['scenario_saving_per_q']:,} / quarter)")

    # ---------------------------------------------------------
    # STEP 8: Validation Metrics & Executive Digests
    # ---------------------------------------------------------
    print("\n[8/8] Computing Human Validation Metrics & Weekly Digests...")

    if human_labels_df is not None:
        test_subset = human_labels_df[
            human_labels_df["human_label"].notna()
            & (human_labels_df["human_label"].astype(str).str.strip() != "")
        ].copy()

        if len(test_subset) > 0:
            test_eval = evaluate_classifier_on_test_set(
                test_df=test_subset,
                predictions_df=classified_df,
                intake_bot_df=tickets_df,
            )

            print(
                f"  Human Validation Accuracy: "
                f"{test_eval.get('accuracy', 'N/A')}, "
                f"Macro-F1: {test_eval.get('macro_f1', 'N/A')}"
            )
        else:
            print("  ⚠ No completed human labels found — evaluation skipped.")
            print("    Label the 50-ticket sheet and re-run to compute metrics.")
    else:
        print("  ⚠ Human labels not loaded — evaluation skipped.")
        print("    Label the 50-ticket sheet and re-run to compute metrics.")
    
    # Generate weekly digests for the last 4 weeks
    recent_weeks = sorted(tickets_df["week_start"].dropna().unique())[-4:]
    print(f"  Generating executive digests for {len(recent_weeks)} weeks with guardrail verification...")
    for w in recent_weeks:
        w_str = str(w)[:10]
        facts = build_weekly_facts(w_str, tickets_df, weekly_df, pairs_df)
        generate_digest(facts, prompt_version="v1", allow_llm=True)
        
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE! ALL ARTIFACTS COMMITTED TO outputs/")
    print("  Ready to launch Streamlit: streamlit run app.py")
    print("=" * 65)


def _create_fallback_classification_cache(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Emergency fallback: copy intake-bot categories into classified_tickets.csv.

    ⚠ WARNING: This means Ollama classification FAILED or was skipped.
    All rows will have status='fallback_seed' and the categories are
    the raw intake-bot names (which may NOT match the canonical taxonomy).
    Re-run with Ollama available to get real classifications.
    """
    print("\n" + "!" * 60)
    print("  ⚠ WARNING: CLASSIFICATION FALLBACK ACTIVE")
    print("  Ollama was unreachable. Using intake-bot categories as-is.")
    print("  These are NOT Gemma classifications. Status = 'fallback_seed'.")
    print("  Re-run without --skip-classify once Ollama is running.")
    print("!" * 60 + "\n")

    cache_path = os.path.join(config.OUTPUTS_DIR, "classified_tickets.csv")
    records = []
    for _, row in tickets_df.iterrows():
        records.append({
            "ticket_id": row["ticket_id"],
            "message_hash": str(hash(str(row["customer_message"])))[:16],
            "category": row.get("category", "Other"),
            "confidence": 0.0,  # Mark 0.0 so it's obvious these are not real
            "model": "fallback",
            "prompt_version": "v1",
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "fallback_seed",
            "attempts": 0,
        })
    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vireo Support Intelligence pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickets for smoke testing")
    parser.add_argument("--skip-embed", action="store_true", help="Skip heavy embedding generation")
    parser.add_argument("--skip-classify", action="store_true", help="Use existing classified_tickets.csv cache")
    args = parser.parse_args()
    
    run_pipeline(limit=args.limit, skip_embed=args.skip_embed, skip_classify=args.skip_classify)
