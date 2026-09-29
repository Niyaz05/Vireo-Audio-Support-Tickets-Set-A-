"""Vireo Support Intelligence — Preprocessing.

Produces a clean, enriched ticket dataset with:
- Timezone corrections for legacy resolved_at (Policy §9)
- Duplicate detection and flagging
- CSAT cleaning (Policy §8)
- SLA breach computation (Policy §3)
- Cost calculations (Policy §4, §5)
- PII masking
- Data quality flags

All calculations are deterministic and unit-tested.
"""

import re
import pandas as pd
import numpy as np
from . import config
from .data_loader import load_tickets, load_products, load_agents, load_orders


def mask_pii(text: str) -> str:
    """Mask PII: phone numbers, emails, long digit strings. Never show customer names."""
    if not isinstance(text, str):
        return text
    # Email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    # Phone numbers (Indian: 10+ digits, with optional +91 prefix)
    text = re.sub(r'(?:\+91[\s-]?)?[6-9]\d{9}', '[PHONE]', text)
    # Long digit strings (>= 6 digits, likely order IDs left in messages are OK)
    text = re.sub(r'\b\d{10,}\b', '[DIGITS]', text)
    return text


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect duplicate tickets across source systems.
    
    Policy §9: "a subset of legacy tickets was re-imported during reconciliation
    and may appear in exports under both source systems."
    
    Strategy: ticket_id appearing in both helpdesk and legacy_fd.
    Keep the helpdesk row (correct timestamps), flag legacy as duplicate.
    """
    # Find ticket_ids that appear in both sources
    dup_mask = df.duplicated(subset=["ticket_id"], keep=False)
    dup_ids = df.loc[dup_mask, "ticket_id"].unique()
    
    # For each duplicate, flag the legacy_fd row
    is_dup_legacy = (df["ticket_id"].isin(dup_ids)) & (df["source_system"] == "legacy_fd")
    
    return is_dup_legacy


def correct_legacy_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Correct legacy resolved_at timestamps from UTC to IST.
    
    Policy §9: "Resolution timestamps for migrated tickets were reconstructed 
    from the legacy event log, which stores UTC."
    
    EDA confirms: 618/618 overlap tickets show exactly +5:30 offset.
    """
    ist_offset = pd.Timedelta(hours=5, minutes=30)
    legacy_mask = df["source_system"] == "legacy_fd"
    
    # Save raw value
    df["resolved_at_raw"] = df["resolved_at"]
    
    # Correct legacy timestamps
    df.loc[legacy_mask, "resolved_at"] = df.loc[legacy_mask, "resolved_at"] + ist_offset
    
    # Flag corrected rows
    df["tz_corrected"] = legacy_mask & df["resolved_at_raw"].notna()
    
    return df


def clean_csat(df: pd.DataFrame) -> pd.DataFrame:
    """Clean CSAT scores.
    
    Policy §8: "A blank score means no response and must be excluded from 
    averages, not treated as zero."
    
    EDA confirms: Legacy uses 0 for no-response (2,083 cases).
    """
    clean = pd.to_numeric(df["csat_score"], errors="coerce")
    clean.loc[clean == 0] = np.nan
    clean.loc[~clean.isin([1, 2, 3, 4, 5])] = np.nan
    df["csat_clean"] = clean
    return df


def compute_sla_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SLA metrics.
    
    Policy §3: "First response is the first reply by a human agent, measured 
    from ticket creation." Targets: chat 15, voice 120, social 240, email 480 min.
    "Every ticket that misses its first-response target automatically issues 
    a store credit of Rs 350 to the customer's account on resolution."
    """
    # First response time in minutes
    df["first_response_minutes"] = (
        (df["first_response_at"] - df["created_at"]).dt.total_seconds() / 60
    ).round(2)
    
    # SLA target by channel
    df["sla_target_min"] = df["channel"].map(config.SLA_TARGET_MINUTES)
    
    # SLA breach: first response later than target
    df["sla_breach"] = df["first_response_minutes"] > df["sla_target_min"]
    
    # Missing first response flag (separate from breach)
    df["missing_first_response"] = df["first_response_at"].isna()
    
    # Policy §3: credit issued on resolution. Only for resolved/closed tickets.
    df["sla_credit_inr"] = 0.0
    resolved_mask = df["status"].isin(config.CLOSURE_STATUSES)
    df.loc[df["sla_breach"] & resolved_mask, "sla_credit_inr"] = config.SLA_CREDIT_INR
    
    return df


def compute_costs(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Compute cost fields.
    
    Policy §4: Contact costs and transfer costs.
    Policy §5: Replacement cost = unit_cost + Rs 340.
    """
    # Contact cost by channel (Policy §4)
    df["contact_cost_inr"] = df["channel"].map(config.CONTACT_COST_INR).fillna(0.0)
    
    # Transfer cost (Policy §4: Rs 305 per transfer)
    df["transfer_cost_inr"] = df["transfers"].fillna(0) * config.TRANSFER_COST_INR
    
    # Replacement cost (Policy §5)
    if "product_sku" in df.columns and products is not None and len(products) > 0:
        sku_col = "sku" if "sku" in products.columns else "product_sku"
        prod_cost = products.set_index(sku_col)["unit_cost_inr"].to_dict()
        df["unit_cost_inr"] = df["product_sku"].map(prod_cost).fillna(df.get("unit_cost_inr", 0.0))
    elif "unit_cost_inr" not in df.columns:
        df["unit_cost_inr"] = 0.0
        
    df["replacement_cost_inr"] = 0.0
    repl_mask = df.get("replacement_issued", pd.Series(index=df.index, data="N")) == "Y"
    df.loc[repl_mask, "replacement_cost_inr"] = (
        df.loc[repl_mask, "unit_cost_inr"] + config.REPLACEMENT_SHIPPING_INR
    )
    
    return df


def compute_handle_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute handle time.
    
    Policy §10: "Handle time: first response to resolution."
    Uses corrected timestamps.
    """
    df["handle_minutes"] = (
        (df["resolved_at"] - df["first_response_at"]).dt.total_seconds() / 60
    ).round(2)
    return df


def compute_week_start(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Monday-start week in IST.
    
    Task spec: "Weeks are Monday-start in IST."
    """
    # W-SUN period means the week ends on Sunday (starts Monday)
    df["week_start"] = df["created_at"].dt.to_period("W-SUN").apply(lambda x: x.start_time)
    return df


def join_orders(df: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Join tickets to orders.
    
    Strategy: Use order_id if available. Fall back to customer_id + product_sku,
    picking the latest order on or before created_at.
    """
    # Direct join on order_id
    has_order = df["order_id"].notna()
    
    # For tickets without order_id, try customer_id + product_sku fallback
    no_order = df[~has_order].copy()
    
    if len(no_order) > 0:
        # Get possible matches
        fallback_matches = no_order.merge(
            orders[["order_id", "customer_id", "sku", "order_date", "lot_code"]],
            left_on=["customer_id", "product_sku"],
            right_on=["customer_id", "sku"],
            how="left",
            suffixes=("", "_order"),
        )
        
        # Filter: order_date on or before created_at
        fallback_matches = fallback_matches[
            fallback_matches["order_date"] <= fallback_matches["created_at"]
        ]
        
        # Pick the latest order for each ticket
        if len(fallback_matches) > 0:
            fallback_matches = fallback_matches.sort_values("order_date", ascending=False)
            best_match = fallback_matches.drop_duplicates(subset=["ticket_id", "source_system"], keep="first")
            
            # Count ambiguous matches (more than one candidate)
            match_counts = fallback_matches.groupby(["ticket_id", "source_system"]).size()
            ambiguous = (match_counts > 1).sum()
            
            # Update the main dataframe
            match_dict = best_match.set_index(["ticket_id", "source_system"])["order_id_order"].to_dict()
            lot_dict = best_match.set_index(["ticket_id", "source_system"])["lot_code"].to_dict()
            
            for idx in no_order.index:
                key = (df.at[idx, "ticket_id"], df.at[idx, "source_system"])
                if key in match_dict:
                    df.at[idx, "order_id_matched"] = match_dict[key]
                    df.at[idx, "lot_code_matched"] = lot_dict.get(key)
            
            df["order_join_method"] = "none"
            df.loc[has_order, "order_join_method"] = "direct"
            df.loc[df["order_id_matched"].notna(), "order_join_method"] = "fallback"
        else:
            df["order_join_method"] = "none"
            df.loc[has_order, "order_join_method"] = "direct"
            ambiguous = 0
    else:
        df["order_join_method"] = "direct"
        ambiguous = 0
    
    # Merge lot_code from orders for direct-join tickets
    order_lot = orders.set_index("order_id")["lot_code"].to_dict()
    df["lot_code"] = df["order_id"].map(order_lot)
    # Fill in fallback matches
    if "lot_code_matched" in df.columns:
        df["lot_code"] = df["lot_code"].fillna(df.get("lot_code_matched"))
    
    # Store join stats
    df.attrs["order_join_direct"] = has_order.sum()
    df.attrs["order_join_fallback"] = df["order_id_matched"].notna().sum() if "order_id_matched" in df.columns else 0
    df.attrs["order_join_none"] = (~has_order).sum() - df.attrs["order_join_fallback"]
    df.attrs["order_join_ambiguous"] = ambiguous if "ambiguous" in dir() else 0
    
    return df


def join_products(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Join product details onto tickets."""
    prod_info = products[["sku", "product_name", "family", "retail_price_inr", "warranty_months"]].copy()
    df = df.merge(prod_info, left_on="product_sku", right_on="sku", how="left", suffixes=("", "_prod"))
    if "sku" in df.columns and "product_sku" in df.columns:
        df.drop(columns=["sku"], inplace=True, errors="ignore")
    return df


def resolve_agent_roster(df: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    """Resolve agent team/tier/site/shift effective at resolved_at.
    
    Policy §7: "one row per assignment with from/to dates"
    Join on agent_id, never on name.
    """
    roster = agents.copy()
    roster["from_date"] = pd.to_datetime(roster["from_date"])
    roster["to_date"] = pd.to_datetime(roster["to_date"])
    
    # Use resolved_at or created_at for attribution
    if "resolved_at" not in df.columns or df["resolved_at"].isna().all():
        date_series = pd.to_datetime(df.get("created_at", pd.Timestamp.now()))
    else:
        date_series = pd.to_datetime(df["resolved_at"].fillna(df.get("created_at", pd.Timestamp.now())))

    roster_by_agent = {}
    for _, r in roster.iterrows():
        roster_by_agent.setdefault(str(r["agent_id"]), []).append(r)
        
    teams, tiers, sites, shifts, effectives, missing = [], [], [], [], [], []
    
    for idx, row in df.iterrows():
        aid = str(row.get("agent_id", ""))
        ticket_dt = date_series.loc[idx]
        assignments = roster_by_agent.get(aid, [])
        if not assignments:
            teams.append(np.nan)
            tiers.append(np.nan)
            sites.append(np.nan)
            shifts.append(np.nan)
            effectives.append(False)
            missing.append(True)
            continue
            
        matched_assign = None
        for a in assignments:
            f_dt = a["from_date"]
            t_dt = a["to_date"]
            if pd.notna(f_dt) and ticket_dt < f_dt:
                continue
            if pd.notna(t_dt) and ticket_dt > t_dt:
                continue
            matched_assign = a
            break
            
        if matched_assign is None:
            # Fallback to latest assignment if outside dates
            matched_assign = sorted(assignments, key=lambda x: str(x.get("from_date", "")))[-1]
            effectives.append(False)
        else:
            effectives.append(True)
            
        teams.append(matched_assign.get("team", np.nan))
        tiers.append(matched_assign.get("tier", np.nan))
        sites.append(matched_assign.get("site", np.nan))
        shifts.append(matched_assign.get("shift", np.nan))
        missing.append(False)
        
    res = df.copy()
    res["agent_team"] = teams
    res["agent_tier"] = tiers
    res["agent_site"] = sites
    res["agent_shift"] = shifts
    res["roster_effective"] = effectives
    res["agent_missing_roster"] = missing
    return res


def build_dq_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Build data quality flags column."""
    flags = []
    for _, row in df.iterrows():
        row_flags = []
        if row.get("is_duplicate", False):
            row_flags.append("duplicate_legacy")
        if row.get("tz_corrected", False):
            row_flags.append("tz_corrected")
        if row.get("missing_first_response", False):
            row_flags.append("missing_first_response")
        if row.get("agent_missing_roster", False):
            row_flags.append("agent_missing_roster")
        if pd.notna(row.get("csat_score")) and row.get("csat_score") == 0:
            row_flags.append("csat_zero_to_nan")
        if pd.notna(row.get("refund_amount_inr")) and row.get("refund_amount_inr", 0) > 0 and row.get("replacement_issued") == "Y":
            row_flags.append("refund_and_replacement")
        if row.get("handle_minutes") is not None and pd.notna(row.get("handle_minutes")) and row.get("handle_minutes", 0) < 0:
            row_flags.append("negative_handle_time")
        flags.append("|".join(row_flags) if row_flags else "")
    df["dq_flags"] = flags
    return df


def preprocess(limit: int = None) -> pd.DataFrame:
    """Run full preprocessing pipeline.
    
    Args:
        limit: If set, only process the first N tickets (for smoke testing).
    
    Returns:
        Clean, enriched DataFrame.
    """
    from .data_loader import load_tickets, load_products, load_agents, load_orders
    
    # Load data
    tickets = load_tickets()
    products = load_products()
    agents = load_agents()
    orders = load_orders()
    
    if limit:
        tickets = tickets.head(limit)
    
    # Step 1: Detect duplicates
    tickets["is_duplicate"] = detect_duplicates(tickets)
    
    # Step 2: Correct legacy timestamps
    tickets = correct_legacy_timestamps(tickets)
    
    # Step 3: Clean CSAT
    tickets = clean_csat(tickets)
    
    # Step 4: Compute week start
    tickets = compute_week_start(tickets)
    
    # Step 5: SLA metrics
    tickets = compute_sla_metrics(tickets)
    
    # Step 6: Join products
    tickets = join_products(tickets, products)
    
    # Step 7: Costs
    tickets = compute_costs(tickets, products)
    
    # Step 8: Handle time
    tickets = compute_handle_time(tickets)
    
    # Step 9: Order joins
    tickets = join_orders(tickets, orders)
    
    # Step 10: Agent roster
    tickets = resolve_agent_roster(tickets, agents)
    
    # Step 11: PII masking for display
    tickets["customer_message_masked"] = tickets["customer_message"].apply(mask_pii)
    tickets["agent_notes_masked"] = tickets["agent_notes"].apply(mask_pii)
    
    # Step 12: Build DQ flags
    tickets = build_dq_flags(tickets)
    
    # Save
    output_path = os.path.join(config.OUTPUTS_DIR, "tickets_clean.csv")
    tickets.to_csv(output_path, index=False)
    
    # Print summary
    total = len(tickets)
    dups = tickets["is_duplicate"].sum()
    print(f"Preprocessing complete: {total} rows, {dups} duplicates flagged")
    print(f"Unique tickets after dedup: {total - dups}")
    print(f"Output saved to {output_path}")
    
    return tickets


import os

if __name__ == "__main__":
    preprocess()
