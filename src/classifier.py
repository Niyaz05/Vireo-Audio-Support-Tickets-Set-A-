"""Vireo Support Intelligence — Taxonomy Discovery & Ticket Classifier.

Uses Gemma via Ollama for:
1. Taxonomy discovery (one-time): sample messages → recurring themes → consolidated taxonomy.
2. Ticket classification: one call per unique message, structured JSON output with enum constraint.

All prompts are in prompts/*.txt. Caches results to outputs/classified_tickets.csv.
"""

import hashlib
import json
import os
import re
import time
import pandas as pd
import numpy as np
import requests
from . import config


def message_hash(text: str) -> str:
    """Hash normalised message for dedup."""
    if not isinstance(text, str):
        return ""
    normalised = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def call_ollama(prompt: str, system: str = "", temperature: float = 0.0,
                max_tokens: int = 200, json_schema: dict = None) -> dict:
    """Call Ollama chat API. Returns dict with response text and metadata."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,  # CRITICAL: Gemma 4 defaults to thinking mode which puts output
                         # in "thinking" field and leaves "content" empty → parse failures.
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    
    # Structured JSON output if schema provided
    if json_schema:
        body["format"] = json_schema
    
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data["message"]["content"],
            "model": data.get("model", config.OLLAMA_MODEL),
            "total_duration": data.get("total_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "ok": True,
        }
    except requests.exceptions.ConnectionError:
        return {
            "content": "",
            "ok": False,
            "error": f"Ollama not running. Start with: ollama serve\nThen pull model: ollama pull {config.OLLAMA_MODEL}",
        }
    except Exception as e:
        return {"content": "", "ok": False, "error": str(e)}


def discover_taxonomy(messages: pd.Series, n_sample: int = 300,
                      batch_size: int = 30) -> list:
    """Discover recurring themes from a sample of messages.
    
    Args:
        messages: Series of customer messages.
        n_sample: Number of messages to sample.
        batch_size: Messages per discovery batch.
    
    Returns:
        List of discovered themes.
    """
    # Stratified sample would need channel/week info; use random for now
    sample = messages.dropna().drop_duplicates().sample(
        min(n_sample, len(messages)), random_state=42
    )
    
    # Load prompt template
    prompt_path = os.path.join(config.PROMPTS_DIR, "discover_taxonomy_v1.txt")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    all_themes = []
    
    for i in range(0, len(sample), batch_size):
        batch = sample.iloc[i:i + batch_size].tolist()
        batch_text = "\n---\n".join(batch)
        prompt = prompt_template.replace("{MESSAGES}", batch_text)
        
        result = call_ollama(prompt, temperature=0.0, max_tokens=500)
        if result["ok"]:
            all_themes.append(result["content"])
            print(f"  Discovery batch {i // batch_size + 1}: OK")
        else:
            print(f"  Discovery batch {i // batch_size + 1}: FAILED - {result.get('error')}")
    
    # Save discovery output
    discovery_path = os.path.join(config.DOCS_DIR, "taxonomy_discovery_raw.md")
    with open(discovery_path, "w") as f:
        f.write("# Taxonomy Discovery — Raw Gemma Output\n\n")
        for i, theme in enumerate(all_themes):
            f.write(f"## Batch {i + 1}\n\n{theme}\n\n")
    
    print(f"Discovery output saved to {discovery_path}")
    return all_themes


def classify_ticket(message: str, product_name: str, intake_category: str,
                    few_shot_examples: list = None, prompt_version: str = "v1") -> dict:
    """Classify a single ticket using Gemma with structured output.
    
    Args:
        message: PII-masked customer message.
        product_name: Product name for context.
        intake_category: Intake-bot category (weak hint).
        few_shot_examples: List of (message, category) tuples for few-shot.
        prompt_version: Prompt version for caching.
    
    Returns:
        Dict with category, confidence, status.
    """
    # Load prompt template
    prompt_path = os.path.join(config.PROMPTS_DIR, f"classify_{prompt_version}.txt")
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(config.PROMPTS_DIR, "classify_v1.txt")
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    # Build few-shot block
    few_shot_block = ""
    if few_shot_examples:
        examples = []
        for ex_msg, ex_cat in few_shot_examples[:config.DEFAULT_FEW_SHOT_K]:
            examples.append(f"Message: {ex_msg[:200]}\nCategory: {ex_cat}")
        few_shot_block = "\n\nExamples:\n" + "\n---\n".join(examples)
    
    # Build prompt
    categories_str = ", ".join(config.TAXONOMY)
    prompt = prompt_template.replace("{MESSAGE}", message[:500])
    prompt = prompt.replace("{PRODUCT}", product_name or "Unknown")
    prompt = prompt.replace("{INTAKE_CATEGORY}", intake_category or "None")
    prompt = prompt.replace("{FEW_SHOT}", few_shot_block)
    prompt = prompt.replace("{CATEGORIES}", categories_str)
    
    # System prompt
    system = "You are a support ticket classifier. Output only valid JSON."
    
    # JSON schema for structured output
    json_schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": config.TAXONOMY,
            },
            "confidence": {
                "type": "number",
            },
        },
        "required": ["category", "confidence"],
    }
    
    result = call_ollama(prompt, system=system, temperature=0.0,
                         max_tokens=250, json_schema=json_schema)
    
    if not result["ok"]:
        return {
            "category": "Other",
            "confidence": 0.0,
            "status": "error",
            "error": result.get("error", "Unknown error"),
        }
    
    # Parse response
    try:
        parsed = json.loads(result["content"])
        category = parsed.get("category", "Other")
        confidence = float(parsed.get("confidence", 0.0))
        
        # Validate category is in taxonomy
        if category not in config.TAXONOMY:
            # Try fuzzy match
            for tax_cat in config.TAXONOMY:
                if category.lower() in tax_cat.lower() or tax_cat.lower() in category.lower():
                    category = tax_cat
                    break
            else:
                category = "Other"
                confidence = max(confidence * 0.5, 0.0)
        
        return {
            "category": category,
            "confidence": confidence,
            "status": "ok",
            "model": result.get("model"),
        }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Safe recovery: try to extract JSON from response
        try:
            match = re.search(r'\{[^}]+\}', result["content"])
            if match:
                parsed = json.loads(match.group())
                return {
                    "category": parsed.get("category", "Other"),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "status": "parse_fail_recovered",
                    "model": result.get("model"),
                }
        except:
            pass
        
        return {
            "category": "Other",
            "confidence": 0.0,
            "status": "parse_fail",
            "raw_response": result["content"][:200],
        }

def classify_ticket_batch(
    tickets: list,
    prompt_version: str = "v1"
) -> list:
    """
    Classify multiple tickets in a single Gemma/Ollama request.

    Each item in `tickets` should contain:
        ticket_id
        message
        product_name
        intake_category

    Returns:
        List of classification dictionaries in the same order as input.
    """

    prompt_path = os.path.join(
        config.PROMPTS_DIR,
        f"classify_{prompt_version}.txt"
    )

    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(
            config.PROMPTS_DIR,
            "classify_v1.txt"
        )

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    categories_str = ", ".join(config.TAXONOMY)

    # Build one combined prompt containing all tickets
    ticket_blocks = []

    for idx, ticket in enumerate(tickets):
        message = str(ticket.get("message", ""))[:500]
        product = ticket.get("product_name", "") or "Unknown"
        intake = ticket.get("intake_category", "") or "None"

        ticket_blocks.append(
            f"""
TICKET_INDEX: {idx}
MESSAGE: {message}
PRODUCT: {product}
INTAKE_CATEGORY: {intake}
"""
        )

    batch_prompt = f"""
You are classifying multiple customer support tickets.

For EACH ticket, return exactly ONE classification.

Allowed categories:
{categories_str}

Return ONLY a valid JSON array.

Each array item must have exactly:
{{
    "ticket_index": integer,
    "category": string,
    "confidence": number
}}

The ticket_index MUST correspond to the TICKET_INDEX provided.

Tickets:
{"---".join(ticket_blocks)}
"""

    system = (
        "You are a support ticket classifier. "
        "Output only valid JSON. "
        "Do not add explanations or markdown."
    )

    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "ticket_index": {
                    "type": "integer"
                },
                "category": {
                    "type": "string",
                    "enum": config.TAXONOMY
                },
                "confidence": {
                    "type": "number"
                }
            },
            "required": [
                "ticket_index",
                "category",
                "confidence"
            ]
        }
    }

    result = call_ollama(
        batch_prompt,
        system=system,
        temperature=0.0,
        max_tokens=max(250 * len(tickets), 500),
        json_schema=json_schema
    )

    if not result["ok"]:
        raise RuntimeError(
            result.get("error", "Ollama batch request failed")
        )

    try:
        parsed = json.loads(result["content"])

        if not isinstance(parsed, list):
            raise ValueError("Gemma batch response is not a JSON array")

        if len(parsed) != len(tickets):
            raise ValueError(
                f"Expected {len(tickets)} results, "
                f"received {len(parsed)}"
            )

        # Map results by ticket index
        indexed_results = {}

        for item in parsed:
            idx = int(item["ticket_index"])
            category = item["category"]
            confidence = float(item["confidence"])

            if idx < 0 or idx >= len(tickets):
                raise ValueError(
                    f"Invalid ticket_index returned by Gemma: {idx}"
                )

            if category not in config.TAXONOMY:
                raise ValueError(
                    f"Invalid category returned by Gemma: {category}"
                )

            indexed_results[idx] = {
                "category": category,
                "confidence": confidence,
                "status": "ok",
                "model": result.get("model")
            }

        # Make sure every ticket has exactly one result
        if len(indexed_results) != len(tickets):
            raise ValueError(
                "Gemma did not return exactly one result per ticket"
            )

        return [
            indexed_results[i]
            for i in range(len(tickets))
        ]

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"Invalid Gemma batch response: {e}"
        )

def classify_all(tickets_df: pd.DataFrame, retriever=None,
                 prompt_version: str = "v1", batch_save_every: int = 50,
                 dev_ids: set = None, test_ids: set = None) -> pd.DataFrame:
    """Classify all tickets with caching, resume, and similarity-based reuse.

    Optimisation strategy (in order of preference):
    1. **Exact cache hit** — message_hash already classified → skip.
    2. **BM25 similarity reuse** — if the top BM25 hit among already-
       classified tickets has score >= 25 AND confidence >= 0.90, copy its
       category (status = 'sim_reuse').  This avoids a Gemma call for
       near-duplicate messages (e.g. same complaint, different order number).
    3. **Gemma call** — classify with the LLM.

    Args:
        tickets_df: DataFrame with ticket_id, customer_message_masked,
                    product_name, category columns.
        retriever: Optional HybridRetriever (used for BM25 similarity only;
                   dense search is NOT called per-ticket).
        prompt_version: Prompt version string.
        batch_save_every: Save cache every N tickets.
        dev_ids: Set of ticket_ids in the dev set (few-shot pool).
        test_ids: Set of ticket_ids in the test set (never used for few-shot).

    Returns:
        DataFrame with classification results.
    """
    cache_path = os.path.join(config.OUTPUTS_DIR, "classified_tickets.csv")

    # Load existing cache
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, dtype=str)
        cached_keys = set(
            cache.apply(lambda r: f"{r['message_hash']}_{r.get('prompt_version', 'v1')}", axis=1)
        )
        print(f"Loaded {len(cache)} cached classifications")
    else:
        cache = pd.DataFrame()
        cached_keys = set()

    # Compute message hashes
    tickets_df = tickets_df.copy()
    tickets_df["message_hash"] = tickets_df["customer_message_masked"].apply(message_hash)

    # Only classify unique messages
    unique_msgs = tickets_df.drop_duplicates(subset=["message_hash"])[
        ["ticket_id", "message_hash", "customer_message_masked", "product_name", "category"]
    ].copy()

    # Skip already cached
    unique_msgs["cache_key"] = unique_msgs["message_hash"] + f"_{prompt_version}"
    to_classify = unique_msgs[~unique_msgs["cache_key"].isin(cached_keys)]

    total_unique = len(unique_msgs)
    total_to_do = len(to_classify)
    print(f"Unique messages: {total_unique}, to classify: {total_to_do}, cached: {len(cached_keys)}")

    if total_to_do == 0:
        stats = {
            "total_classified": 0,
            "cache_hits": total_unique,
            "calls_made": 0,
            "total_time_seconds": 0.0,
            "tickets_per_minute": 0.0,
            "model": config.OLLAMA_MODEL,
            "prompt_version": prompt_version,
            "note": "No new tickets required classification — all results came from cache"
        }

        stats_path = os.path.join(config.OUTPUTS_DIR, "run_stats.json")

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  All {total_unique} unique tickets already cached — no Ollama calls needed.")

        return pd.read_csv(cache_path, dtype=str)

    # --- Build a lookup of already-classified texts for similarity reuse ---
    classified_texts = {}  # message_hash -> (text, category, confidence)
    if len(cache) > 0:
        for _, r in cache.iterrows():
            if r.get("status") == "ok" and float(r.get("confidence", 0)) >= 0.90:
                # We need the text; find it in tickets_df
                match = tickets_df[tickets_df["message_hash"] == r["message_hash"]]
                if len(match) > 0:
                    classified_texts[r["message_hash"]] = (
                        match.iloc[0]["customer_message_masked"],
                        r["category"],
                        float(r["confidence"]),
                    )

    # Pre-build a BM25 index of already-classified texts for similarity reuse
    sim_reuse_bm25 = None
    sim_reuse_keys = []
    if classified_texts and retriever:
        from rank_bm25 import BM25Okapi
        from .retrieval import tokenize
        sim_texts = []
        for mh, (text, cat, conf) in classified_texts.items():
            sim_texts.append(text)
            sim_reuse_keys.append(mh)
        tokenized = [tokenize(t) for t in sim_texts]
        if tokenized:
            sim_reuse_bm25 = BM25Okapi(tokenized)
            print(f"Built similarity-reuse BM25 index with {len(sim_texts)} high-confidence entries")

        # --- Classify ---
    results = []
    start_time = time.time()
    calls_made = 0
    sim_reused = 0

    # True Gemma batch size
    GEMMA_BATCH_SIZE = 8

    # Tickets that survive BM25 reuse and therefore need Gemma
    gemma_queue = []

    for _, row in to_classify.iterrows():
        msg_text = row["customer_message_masked"]

        # --- Strategy 2: BM25 similarity reuse ---
        reused = False

        if sim_reuse_bm25 is not None and len(sim_reuse_keys) > 0:
            from .retrieval import tokenize

            query_tokens = tokenize(msg_text)
            scores = sim_reuse_bm25.get_scores(query_tokens)

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            # High lexical overlap → reuse existing classification
            if best_score >= 25.0:
                donor_hash = sim_reuse_keys[best_idx]
                donor_text, donor_cat, donor_conf = classified_texts[donor_hash]

                results.append({
                    "ticket_id": row["ticket_id"],
                    "message_hash": row["message_hash"],
                    "category": donor_cat,
                    "confidence": round(donor_conf * 0.95, 2),
                    "model": config.OLLAMA_MODEL,
                    "prompt_version": prompt_version,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "status": "sim_reuse",
                    "attempts": 0,
                })

                sim_reused += 1
                reused = True

        # If BM25 could not safely reuse a classification,
        # put the ticket into the Gemma queue.
        if not reused:
            gemma_queue.append(row)

    print(
        f"  BM25 similarity reuse: {sim_reused} tickets"
    )
    print(
        f"  Tickets requiring Gemma: {len(gemma_queue)}"
    )
    print(
        f"  Gemma batch size: {GEMMA_BATCH_SIZE}"
    )

    # ---------------------------------------------------------
    # Strategy 3: TRUE BATCHED GEMMA CLASSIFICATION
    # ---------------------------------------------------------

    for batch_start in range(0, len(gemma_queue), GEMMA_BATCH_SIZE):

        batch_rows = gemma_queue[
            batch_start:batch_start + GEMMA_BATCH_SIZE
        ]

        batch_number = (
            batch_start // GEMMA_BATCH_SIZE
        ) + 1

        total_batches = (
            (len(gemma_queue) + GEMMA_BATCH_SIZE - 1)
            // GEMMA_BATCH_SIZE
        )

        print(
            f"  Gemma batch {batch_number}/{total_batches} "
            f"({len(batch_rows)} tickets)..."
        )

        batch_tickets = []

        for row in batch_rows:
            batch_tickets.append({
                "ticket_id": row["ticket_id"],
                "message": row["customer_message_masked"],
                "product_name": row.get("product_name", ""),
                "intake_category": row["category"],
            })

        try:
            # ONE Ollama request for the entire batch
            batch_results = classify_ticket_batch(
                tickets=batch_tickets,
                prompt_version=prompt_version,
            )

            if len(batch_results) != len(batch_rows):
                raise RuntimeError(
                    f"Expected {len(batch_rows)} results, "
                    f"received {len(batch_results)}"
                )

            # Attach ticket metadata to each Gemma result
            for row, result in zip(batch_rows, batch_results):

                result["ticket_id"] = row["ticket_id"]
                result["message_hash"] = row["message_hash"]
                result["prompt_version"] = prompt_version
                result["timestamp"] = pd.Timestamp.now().isoformat()
                result["attempts"] = 1

                results.append(result)

                # Add high-confidence Gemma results to
                # the similarity-reuse pool.
                if (
                    result.get("status") == "ok"
                    and float(result.get("confidence", 0)) >= 0.90
                ):
                    mh = row["message_hash"]

                    classified_texts[mh] = (
                        row["customer_message_masked"],
                        result["category"],
                        float(result["confidence"]),
                    )

            # IMPORTANT:
            # One batch = one Ollama/Gemma request
            calls_made += 1

        except Exception as e:

            print(
                f"  ⚠ Gemma batch {batch_number} failed: {e}"
            )

            print(
                "  → Falling back to individual classification "
                "for this batch."
            )

            # Safety fallback:
            # If batch classification fails, use the original
            # single-ticket classifier.
            for row in batch_rows:

                result = classify_ticket(
                    message=row["customer_message_masked"],
                    product_name=row.get("product_name", ""),
                    intake_category=row["category"],
                    few_shot_examples=None,
                    prompt_version=prompt_version,
                )

                result["ticket_id"] = row["ticket_id"]
                result["message_hash"] = row["message_hash"]
                result["prompt_version"] = prompt_version
                result["timestamp"] = pd.Timestamp.now().isoformat()
                result["attempts"] = 1

                results.append(result)

                # Add successful high-confidence result to
                # similarity reuse pool.
                if (
                    result.get("status") == "ok"
                    and float(result.get("confidence", 0)) >= 0.90
                ):
                    mh = row["message_hash"]

                    classified_texts[mh] = (
                        row["customer_message_masked"],
                        result["category"],
                        float(result["confidence"]),
                    )

                # This is an individual fallback call.
                calls_made += 1

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        processed_gemma = min(
            batch_start + len(batch_rows),
            len(gemma_queue),
        )

        elapsed = time.time() - start_time

        rate = (
            processed_gemma / elapsed * 60
            if elapsed > 0
            else 0
        )

        remaining = len(gemma_queue) - processed_gemma

        eta_min = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"    Progress: "
            f"{processed_gemma}/{len(gemma_queue)} "
            f"Gemma tickets | "
            f"{rate:.0f} tickets/min | "
            f"Ollama calls: {calls_made} | "
            f"BM25 reuse: {sim_reused} | "
            f"ETA: {eta_min:.1f} min"
        )

        # Save periodically
        if (
            (batch_number % max(1, batch_save_every // GEMMA_BATCH_SIZE) == 0)
            or batch_start + len(batch_rows) >= len(gemma_queue)
        ):
            _save_results(
                cache,
                results,
                cache_path
            )
    # Final save
    if results:
        _save_results(cache, results, cache_path)

    # Run stats
    elapsed = time.time() - start_time
    stats = {
        "total_classified": len(results),
        "llm_calls": calls_made,
        "sim_reused": sim_reused,
        "cache_hits": len(cached_keys),
        "total_time_seconds": round(elapsed, 1),
        "tickets_per_minute": round(len(results) / max(elapsed / 60, 0.01), 1),
        "model": config.OLLAMA_MODEL,
        "prompt_version": prompt_version,
        "note": "Local inference via Ollama — no marginal API cost",
    }
    stats_path = os.path.join(config.OUTPUTS_DIR, "run_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Run stats saved to {stats_path}")
    print(f"  LLM calls: {calls_made}, sim-reuse: {sim_reused}, "
          f"savings: {sim_reused / max(len(results), 1) * 100:.0f}%")

    # Reload full cache
    return pd.read_csv(cache_path, dtype=str)


def _save_results(cache_df, new_results, cache_path):
    """Save classification results to cache file."""
    new_df = pd.DataFrame(new_results)
    cols = ["ticket_id", "message_hash", "category", "confidence", "model",
            "prompt_version", "timestamp", "status", "attempts"]
    available_cols = [c for c in cols if c in new_df.columns]
    new_df = new_df[available_cols]

    if len(cache_df) > 0:
        combined = pd.concat([cache_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(cache_path, index=False)

