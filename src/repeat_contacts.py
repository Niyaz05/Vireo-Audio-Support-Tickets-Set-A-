"""Vireo Support Intelligence — Repeat Contact Detection.

Transparent, rule-based repeat-contact detection as defined in Policy §10:
"A ticket is resolved at first contact if the same customer does not contact
again about the same issue within 30 days of resolution."

Rules:
- For each resolved/closed ticket A, look at later tickets B from same customer_id
  created within 30 days after A's resolved_at (corrected timestamps).
- Score B as a repeat of A from independent evidence:
  * same order_id (+3)
  * same product_sku where order_id is missing (+2)
  * same classified category (+1)
  * hybrid text-similarity above threshold (+2)
  * phrases like "already told / again / still not fixed" (+2)
- Minimum evidence score for strict (5) and loose (3) definitions.
- Same customer alone is never a repeat.
- Right-censor last 30 days.
"""

import re
import os
import pandas as pd
import numpy as np
from . import config

# Repeat-indicator phrases
REPEAT_PHRASES = [
    r'already told',
    r'again',
    r'still not fixed',
    r'still not resolved',
    r'still not working',
    r'second time',
    r'third time',
    r'multiple times',
    r'follow\s*up',
    r'same issue',
    r'same problem',
    r'not resolved',
    r'earlier complaint',
    r'previous ticket',
    r'contacted before',
    r'called before',
    r'wrote before',
    r'no response',
    r'no reply',
    r'waiting since',
]
REPEAT_PATTERN = re.compile('|'.join(REPEAT_PHRASES), re.IGNORECASE)


def has_repeat_phrases(text: str) -> bool:
    """Check if message contains repeat-indicator phrases."""
    if not isinstance(text, str):
        return False
    return bool(REPEAT_PATTERN.search(text))


def compute_repeat_contacts(tickets_df: pd.DataFrame, 
                            text_sim_func=None,
                            strict_threshold: int = 5,
                            loose_threshold: int = 3) -> pd.DataFrame:
    """Detect repeat contacts.
    
    Args:
        tickets_df: Clean tickets DataFrame.
        text_sim_func: Optional function(text_a, text_b) -> similarity score (0-1).
        strict_threshold: Minimum evidence score for strict definition.
        loose_threshold: Minimum evidence score for loose definition.
    
    Returns:
        DataFrame of repeat contact pairs with evidence.
    """
    # Only resolved/closed tickets
    resolved = tickets_df[
        (tickets_df["status"].isin(config.CLOSURE_STATUSES)) &
        (~tickets_df["is_duplicate"]) &
        (tickets_df["resolved_at"].notna())
    ].copy()
    
    # Right-censor: exclude tickets resolved less than 30 days before data end
    data_end = tickets_df["created_at"].max()
    censor_date = data_end - pd.Timedelta(days=config.REPEAT_WINDOW_DAYS)
    eligible = resolved[resolved["resolved_at"] <= censor_date].copy()
    censored_count = len(resolved) - len(eligible)
    
    print(f"Repeat contact detection:")
    print(f"  Resolved/closed tickets: {len(resolved)}")
    print(f"  Eligible (not censored): {len(eligible)}")
    print(f"  Right-censored (last 30 days): {censored_count}")
    
    # All tickets (including open/pending) can be the "repeat" ticket B
    all_tickets = tickets_df[~tickets_df["is_duplicate"]].copy()
    
    pairs = []
    
    for _, ticket_a in eligible.iterrows():
        # Find later tickets B from same customer within 30 days
        mask = (
            (all_tickets["customer_id"] == ticket_a["customer_id"]) &
            (all_tickets["created_at"] > ticket_a["resolved_at"]) &
            (all_tickets["created_at"] <= ticket_a["resolved_at"] + pd.Timedelta(days=config.REPEAT_WINDOW_DAYS)) &
            (all_tickets["ticket_id"] != ticket_a["ticket_id"])
        )
        
        candidates = all_tickets[mask]
        
        for _, ticket_b in candidates.iterrows():
            evidence_score = 0
            match_reasons = []
            
            # Same order_id (+3)
            if (pd.notna(ticket_a.get("order_id")) and pd.notna(ticket_b.get("order_id")) and
                    ticket_a["order_id"] == ticket_b["order_id"]):
                evidence_score += 3
                match_reasons.append("same_order")
            
            # Same product_sku (+2, only when order_id missing)
            elif (pd.isna(ticket_a.get("order_id")) or pd.isna(ticket_b.get("order_id"))):
                if ticket_a["product_sku"] == ticket_b["product_sku"]:
                    evidence_score += 2
                    match_reasons.append("same_sku")
            
            # Same classified category (+1)
            cat_col = "classified_category" if "classified_category" in ticket_a.index else "category"
            if ticket_a.get(cat_col) == ticket_b.get(cat_col):
                evidence_score += 1
                match_reasons.append("same_category")
            
            # Text similarity (+2)
            if text_sim_func:
                try:
                    sim = text_sim_func(
                        ticket_a.get("customer_message_masked", ""),
                        ticket_b.get("customer_message_masked", "")
                    )
                    if sim > 0.7:
                        evidence_score += 2
                        match_reasons.append(f"text_sim={sim:.2f}")
                except:
                    pass
            
            # Repeat phrases (+2)
            if has_repeat_phrases(ticket_b.get("customer_message", "")):
                evidence_score += 2
                match_reasons.append("repeat_phrases")
            
            if evidence_score >= loose_threshold:
                pairs.append({
                    "ticket_a_id": ticket_a["ticket_id"],
                    "ticket_b_id": ticket_b["ticket_id"],
                    "customer_id": ticket_a["customer_id"],
                    "days_between": (ticket_b["created_at"] - ticket_a["resolved_at"]).days,
                    "evidence_score": evidence_score,
                    "match_reason": "|".join(match_reasons),
                    "is_strict": evidence_score >= strict_threshold,
                    "is_loose": evidence_score >= loose_threshold,
                    "repeat_channel": ticket_b["channel"],
                    "repeat_cost_inr": ticket_b.get("contact_cost_inr", 0),
                    "ticket_a_transfers": ticket_a.get("transfers", 0),
                    "ticket_a_status": ticket_a["status"],
                    "ticket_a_channel": ticket_a["channel"],
                })
    
    pairs_df = pd.DataFrame(pairs)
    
    if len(pairs_df) > 0:
        # Compute rates
        n_eligible = len(eligible)
        strict_repeats = pairs_df["is_strict"].sum()
        loose_repeats = pairs_df["is_loose"].sum()
        # Unique ticket_a with at least one repeat
        strict_unique = pairs_df[pairs_df["is_strict"]]["ticket_a_id"].nunique()
        loose_unique = pairs_df[pairs_df["is_loose"]]["ticket_a_id"].nunique()
        
        print(f"\n  Strict repeats ({strict_threshold}+ evidence): {strict_repeats} pairs, {strict_unique} unique tickets ({strict_unique/n_eligible*100:.1f}%)")
        print(f"  Loose repeats ({loose_threshold}+ evidence): {loose_repeats} pairs, {loose_unique} unique tickets ({loose_unique/n_eligible*100:.1f}%)")
        print(f"  Total repeat cost (strict): Rs {pairs_df[pairs_df['is_strict']]['repeat_cost_inr'].sum():,.0f}")
        print(f"  Total repeat cost (loose): Rs {pairs_df[pairs_df['is_loose']]['repeat_cost_inr'].sum():,.0f}")
    
    # Save
    output_path = os.path.join(config.OUTPUTS_DIR, "repeat_contacts.csv")
    pairs_df.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    
    # Store metadata
    pairs_df.attrs["n_eligible"] = len(eligible)
    pairs_df.attrs["n_censored"] = censored_count
    pairs_df.attrs["strict_threshold"] = strict_threshold
    pairs_df.attrs["loose_threshold"] = loose_threshold
    
    return pairs_df


def analyse_colleague_claim(tickets_df: pd.DataFrame, pairs_df: pd.DataFrame) -> dict:
    """Test the 'I already told your colleague' claim from the data.
    
    Compare transfers>0 tickets against the rest: their repeat rate,
    rate of repeat-phrase messages, and channel distribution.
    """
    resolved = tickets_df[
        (tickets_df["status"].isin(config.CLOSURE_STATUSES)) &
        (~tickets_df["is_duplicate"]) &
        (tickets_df["resolved_at"].notna())
    ]
    
    # Tickets with transfers > 0
    transferred = resolved[resolved["transfers"] > 0]
    not_transferred = resolved[resolved["transfers"] == 0]
    
    # Repeat rates
    if len(pairs_df) > 0:
        strict_a_ids = set(pairs_df[pairs_df["is_strict"]]["ticket_a_id"])
        
        trans_repeat = len(set(transferred["ticket_id"]) & strict_a_ids)
        no_trans_repeat = len(set(not_transferred["ticket_id"]) & strict_a_ids)
        
        trans_rate = trans_repeat / max(len(transferred), 1) * 100
        no_trans_rate = no_trans_repeat / max(len(not_transferred), 1) * 100
    else:
        trans_rate = 0
        no_trans_rate = 0
    
    # Repeat-phrase rates
    trans_phrases = transferred["customer_message"].apply(has_repeat_phrases).sum()
    no_trans_phrases = not_transferred["customer_message"].apply(has_repeat_phrases).sum()
    
    # Channel distribution
    trans_chat_share = (transferred["channel"] == "chat").mean() * 100
    
    result = {
        "transferred_tickets": len(transferred),
        "not_transferred_tickets": len(not_transferred),
        "transferred_repeat_rate_pct": round(trans_rate, 1),
        "not_transferred_repeat_rate_pct": round(no_trans_rate, 1),
        "transferred_phrase_count": int(trans_phrases),
        "transferred_phrase_rate_pct": round(trans_phrases / max(len(transferred), 1) * 100, 1),
        "not_transferred_phrase_count": int(no_trans_phrases),
        "not_transferred_phrase_rate_pct": round(no_trans_phrases / max(len(not_transferred), 1) * 100, 1),
        "transferred_chat_share_pct": round(trans_chat_share, 1),
    }
    
    # Also check: is repeat more likely after auto-closed or fast resolution?
    auto_closed = resolved[resolved["status"] == "closed"]
    fast_resolved = resolved[
        (resolved["handle_minutes"].notna()) & (resolved["handle_minutes"] < 10)
    ]
    
    if len(pairs_df) > 0:
        ac_repeat = len(set(auto_closed["ticket_id"]) & strict_a_ids)
        fast_repeat = len(set(fast_resolved["ticket_id"]) & strict_a_ids)
        result["auto_closed_repeat_rate_pct"] = round(ac_repeat / max(len(auto_closed), 1) * 100, 1)
        result["fast_resolved_repeat_rate_pct"] = round(fast_repeat / max(len(fast_resolved), 1) * 100, 1)
    
    return result
