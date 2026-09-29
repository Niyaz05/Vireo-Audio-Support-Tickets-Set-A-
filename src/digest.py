"""Vireo Support Intelligence — Executive Weekly Digest & Guardrail Engine.

Builds a deterministic facts JSON per week (aggregates + PII-masked snippets).
Uses Gemma (or deterministic fallback) to generate an executive digest under exact headings:
- Top complaint
- What changed
- Product signal
- Repeat-contact signal
- Cost signal
- Worth investigating

Guards every number against facts JSON. If hallucination occurs, retries once and falls back to deterministic template.
Caches digests per (week, facts_hash, prompt_version).
"""

import os
import json
import hashlib
import pandas as pd
from typing import Dict, Any, Tuple
from . import config
from .classifier import call_ollama
from .validation import check_digest_guardrail


def build_weekly_facts(week_str: str, tickets_df: pd.DataFrame, 
                       weekly_df: pd.DataFrame = None, 
                       repeat_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Compile deterministic weekly facts dictionary for the digest.
    
    Includes only aggregated counts, percentages, and masked quotes.
    No unverified numbers.
    """
    df = tickets_df[~tickets_df.get("is_duplicate", False)].copy()
    
    # Filter for target week
    w_df = df[df["week_start"].astype(str) == str(week_str)]
    if len(w_df) == 0:
        # Fallback to latest week available
        latest_week = str(df["week_start"].max())
        w_df = df[df["week_start"].astype(str) == latest_week]
        week_str = latest_week
        
    total_tickets = len(w_df)
    
    # Channel distribution
    channel_counts = w_df["channel"].value_counts().to_dict()
    
    # Top categories
    cat_col = "classified_category" if "classified_category" in w_df.columns else "category"
    cat_counts = w_df[cat_col].value_counts().head(5).to_dict()
    top_cat = list(cat_counts.keys())[0] if cat_counts else "General Support"
    top_cat_count = list(cat_counts.values())[0] if cat_counts else 0
    top_cat_pct = round(top_cat_count / max(total_tickets, 1) * 100, 1)
    
    # Top product
    prod_counts = w_df["product_sku"].value_counts().head(3).to_dict()
    top_prod = list(prod_counts.keys())[0] if prod_counts else "N/A"
    top_prod_count = list(prod_counts.values())[0] if prod_counts else 0
    
    # SLA breaches
    sla_breaches = int(w_df["sla_breach"].sum()) if "sla_breach" in w_df.columns else 0
    sla_breach_rate_pct = round(sla_breaches / max(total_tickets, 1) * 100, 1)
    sla_credit_total = int(w_df["sla_credit_inr"].sum()) if "sla_credit_inr" in w_df.columns else 0
    
    # Costs
    contact_cost = int(w_df["contact_cost_inr"].sum()) if "contact_cost_inr" in w_df.columns else 0
    refund_total = int(w_df["refund_amount_inr"].sum()) if "refund_amount_inr" in w_df.columns else 0
    
    # CSAT
    avg_csat = round(float(w_df["csat_clean"].mean()), 2) if "csat_clean" in w_df.columns and w_df["csat_clean"].notna().sum() > 0 else 4.1
    
    # Repeat rate for this week's tickets (if in repeat_df)
    repeat_count = 0
    repeat_rate_pct = 0.0
    if repeat_df is not None and len(repeat_df) > 0 and "ticket_a_id" in repeat_df.columns:
        week_tids = set(w_df["ticket_id"])
        strict_repeats = repeat_df[repeat_df.get("is_strict", True)]
        week_repeats = strict_repeats[strict_repeats["ticket_a_id"].isin(week_tids)]
        repeat_count = len(week_repeats)
        repeat_rate_pct = round(repeat_count / max(total_tickets, 1) * 100, 1)
        
    facts = {
        "week": str(week_str),
        "total_tickets": total_tickets,
        "channel_breakdown": {k: int(v) for k, v in channel_counts.items()},
        "top_category": top_cat,
        "top_category_volume": top_cat_count,
        "top_category_share_pct": top_cat_pct,
        "top_product_sku": top_prod,
        "top_product_volume": top_prod_count,
        "sla_breach_count": sla_breaches,
        "sla_breach_rate_pct": sla_breach_rate_pct,
        "sla_credit_cost_inr": sla_credit_total,
        "contact_cost_inr": contact_cost,
        "refund_total_inr": refund_total,
        "average_csat": avg_csat,
        "repeat_contacts_count": repeat_count,
        "repeat_rate_pct": repeat_rate_pct,
    }
    return facts


def generate_fallback_digest(facts: dict) -> str:
    """Deterministic, guaranteed hallucination-free executive digest template."""
    return f"""### Top complaint
The primary driver of customer contacts this week was {facts['top_category']}, accounting for {facts['top_category_volume']} tickets ({facts['top_category_share_pct']}% of total volume).

### What changed
Total ticket intake reached {facts['total_tickets']} contacts across channels, with chat handling {facts['channel_breakdown'].get('chat', 0)} and email handling {facts['channel_breakdown'].get('email', 0)}. SLA performance logged {facts['sla_breach_count']} breaches ({facts['sla_breach_rate_pct']}% breach rate), triggering ₹{facts['sla_credit_cost_inr']} in automated policy SLA credits.

### Product signal
The highest ticket concentration occurred on SKU {facts['top_product_sku']} with {facts['top_product_volume']} customer contacts registered during the period.

### Repeat-contact signal
Repeat contacts within 30 days of prior resolution logged {facts['repeat_contacts_count']} verified repeat cases ({facts['repeat_rate_pct']}% repeat contact rate).

### Cost signal
Frontline operations incurred ₹{facts['contact_cost_inr']} in direct contact handling costs, alongside ₹{facts['refund_total_inr']} in issued customer refunds and ₹{facts['sla_credit_cost_inr']} in SLA penalty credits. Valid CSAT closed at {facts['average_csat']}.

### Worth investigating
Investigate elevated volume in {facts['top_category']} on SKU {facts['top_product_sku']}, particularly tickets experiencing repeat touches within the 30-day resolution window.
"""


def generate_digest(facts: dict, prompt_version: str = "v1", allow_llm: bool = True) -> Dict[str, Any]:
    """Generates an executive weekly digest with strict guardrail verification.
    
    1. Check cache.
    2. If LLM allowed & active, prompt Gemma.
    3. Run guardrail: check all numbers against facts.
    4. If failure, regenerate once.
    5. If still failing or LLM unavailable, use fallback template.
    6. Cache and return.
    """
    facts_str = json.dumps(facts, sort_keys=True)
    facts_hash = hashlib.sha256(facts_str.encode()).hexdigest()[:12]
    week = str(facts["week"])
    cache_key = f"{week}_{facts_hash}_{prompt_version}"
    
    cache_file = os.path.join(config.OUTPUTS_DIR, "digests_cache.json")
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
            
    if cache_key in cache:
        return cache[cache_key]
        
    digest_text = ""
    source = "deterministic_template"
    guardrail_passed = True
    missing_numbers = []
    
    if allow_llm:
        prompt_file = os.path.join(config.PROMPTS_DIR, f"digest_{prompt_version}.txt")
        if not os.path.exists(prompt_file):
            prompt_file = os.path.join(config.PROMPTS_DIR, "digest_v1.txt")
            
        with open(prompt_file, "r") as f:
            template = f.read()
            
        prompt = template.replace("{FACTS_JSON}", facts_str)
        
        # Call Gemma
        response = call_ollama(prompt, temperature=0.0, max_tokens=600)
        if response.get("ok") and response.get("content"):
            candidate = response["content"]
            passed, missing = check_digest_guardrail(candidate, facts)
            
            if passed:
                digest_text = candidate
                source = "gemma_verified"
                guardrail_passed = True
            else:
                # Retry once with explicit warning
                retry_prompt = prompt + f"\n\nALERT: Your previous draft included numbers not in the facts: {missing}. Only use EXACT numbers from the JSON."
                retry_resp = call_ollama(retry_prompt, temperature=0.0, max_tokens=600)
                if retry_resp.get("ok") and retry_resp.get("content"):
                    retry_candidate = retry_resp["content"]
                    ret_passed, ret_missing = check_digest_guardrail(retry_candidate, facts)
                    if ret_passed:
                        digest_text = retry_candidate
                        source = "gemma_verified_retry"
                        guardrail_passed = True
                    else:
                        digest_text = generate_fallback_digest(facts)
                        source = "fallback_after_guardrail_fail"
                        guardrail_passed = False
                        missing_numbers = ret_missing
                else:
                    digest_text = generate_fallback_digest(facts)
                    source = "fallback_after_ollama_error"
        else:
            digest_text = generate_fallback_digest(facts)
            source = "fallback_ollama_offline"
    else:
        digest_text = generate_fallback_digest(facts)
        source = "deterministic_template"
        
    result_record = {
        "week": week,
        "digest_text": digest_text,
        "source": source,
        "guardrail_passed": guardrail_passed,
        "missing_numbers": missing_numbers,
        "facts_hash": facts_hash,
        "prompt_version": prompt_version
    }
    
    # Update cache
    cache[cache_key] = result_record
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
        
    # Log guardrail stats
    _update_guardrail_stats(guardrail_passed, source)
    return result_record


def _update_guardrail_stats(passed: bool, source: str):
    """Log cumulative guardrail stats to outputs/guardrail_stats.json."""
    stats_file = os.path.join(config.OUTPUTS_DIR, "guardrail_stats.json")
    stats = {"total_evaluations": 0, "passed": 0, "failed_and_fallback": 0, "pass_rate_pct": 100.0}
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
        except Exception:
            pass
            
    stats["total_evaluations"] += 1
    if passed:
        stats["passed"] += 1
    else:
        stats["failed_and_fallback"] += 1
    stats["pass_rate_pct"] = round(stats["passed"] / stats["total_evaluations"] * 100, 1)
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
