"""Vireo Support Intelligence — Analytics Engine.

Computes weekly metrics, agent leaderboard, and breach analysis.
All from deterministic Python calculations.
"""

import pandas as pd
import numpy as np
from . import config


def compute_weekly_metrics(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly metrics for all dimensions.
    
    Excludes duplicates. Uses is_duplicate flag.
    """
    # Filter to non-duplicate tickets
    df = tickets_df[~tickets_df["is_duplicate"]].copy()
    
    weeks = df.groupby("week_start")
    
    records = []
    all_weeks = sorted(df["week_start"].dropna().unique())
    
    for week in all_weeks:
        w = df[df["week_start"] == week]
        resolved = w[w["status"].isin(config.CLOSURE_STATUSES)]
        
        record = {
            "week_start": week,
            "total_tickets": len(w),
            # Volume by channel
            "chat_tickets": (w["channel"] == "chat").sum(),
            "email_tickets": (w["channel"] == "email").sum(),
            "voice_tickets": (w["channel"] == "voice").sum(),
            "social_tickets": (w["channel"] == "social").sum(),
            # Status
            "resolved_count": (w["status"] == "resolved").sum(),
            "closed_count": (w["status"] == "closed").sum(),
            "open_count": (w["status"] == "open").sum(),
            "pending_count": (w["status"] == "pending").sum(),
            # SLA
            "sla_breaches": w["sla_breach"].sum(),
            "sla_breach_rate": w["sla_breach"].mean() if len(w) > 0 else 0,
            "sla_credit_total": w["sla_credit_inr"].sum(),
            # Transfers
            "total_transfers": w["transfers"].sum(),
            "transfer_cost": w["transfer_cost_inr"].sum(),
            "tickets_with_transfer": (w["transfers"] > 0).sum(),
            # Costs
            "contact_cost_total": w["contact_cost_inr"].sum(),
            "refund_total": w["refund_amount_inr"].sum(),
            "replacement_cost_total": w["replacement_cost_inr"].sum(),
            # CSAT (Policy §8: exclude zeros/NaN)
            "avg_csat": w["csat_clean"].mean() if w["csat_clean"].notna().sum() > 0 else np.nan,
            "csat_responses": w["csat_clean"].notna().sum(),
            # Replacements
            "replacements_issued": (w["replacement_issued"] == "Y").sum(),
        }
        
        # Category distribution
        cat_col = "classified_category" if "classified_category" in w.columns else "category"
        for cat in config.TAXONOMY:
            record[f"cat_{cat}"] = (w[cat_col] == cat).sum()
        
        records.append(record)
    
    metrics_df = pd.DataFrame(records)
    
    # Compute WoW changes
    if len(metrics_df) > 1:
        metrics_df["wow_volume_change"] = metrics_df["total_tickets"].diff()
        metrics_df["wow_volume_pct"] = metrics_df["total_tickets"].pct_change() * 100
        metrics_df["wow_breach_rate_change"] = metrics_df["sla_breach_rate"].diff()
    
    # Save
    output_path = f"{config.OUTPUTS_DIR}/weekly_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Weekly metrics saved to {output_path}")
    
    return metrics_df


def detect_emerging_categories(weekly_df: pd.DataFrame, n_consecutive: int = 3,
                               spike_threshold: float = 1.5, min_volume: int = 5) -> list:
    """Detect emerging complaint categories.
    
    A category is emerging if:
    - Rising for n_consecutive weeks, OR
    - A spike above rolling baseline with minimum volume guard.
    """
    emerging = []
    cat_cols = [c for c in weekly_df.columns if c.startswith("cat_")]
    
    for col in cat_cols:
        cat_name = col.replace("cat_", "")
        series = weekly_df[col].fillna(0)
        
        if len(series) < n_consecutive + 1:
            continue
        
        # Check consecutive rise
        diffs = series.diff().iloc[1:]
        for i in range(len(diffs) - n_consecutive + 1):
            window = diffs.iloc[i:i + n_consecutive]
            if (window > 0).all() and series.iloc[i + n_consecutive] >= min_volume:
                emerging.append({
                    "category": cat_name,
                    "type": "consecutive_rise",
                    "weeks": n_consecutive,
                    "latest_count": int(series.iloc[i + n_consecutive]),
                    "start_week": str(weekly_df.iloc[i + 1]["week_start"]),
                })
        
        # Check spike
        rolling_mean = series.rolling(4, min_periods=2).mean()
        for i in range(4, len(series)):
            if series.iloc[i] >= min_volume and rolling_mean.iloc[i - 1] > 0:
                ratio = series.iloc[i] / rolling_mean.iloc[i - 1]
                if ratio >= spike_threshold:
                    emerging.append({
                        "category": cat_name,
                        "type": "spike",
                        "ratio": round(ratio, 2),
                        "count": int(series.iloc[i]),
                        "baseline": round(rolling_mean.iloc[i - 1], 1),
                        "week": str(weekly_df.iloc[i]["week_start"]),
                    })
    
    return emerging


def compute_breach_by_hour_shift(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Analyse SLA breaches by hour of day, shift, and channel.
    
    Policy §2: "Overnight coverage is provided by the Indore night shift"
    Policy §3: "Breaches are reported against the resolving agent"
    Test: are breaches a staffing pattern rather than an agent one?
    """
    df = tickets_df[~tickets_df["is_duplicate"]].copy()
    df["created_hour"] = df["created_at"].dt.hour
    
    # Map hour to shift
    def hour_to_shift(h):
        if 6 <= h < 14:
            return "Morning"
        elif 14 <= h < 22:
            return "Day"
        else:
            return "Night"
    
    df["ticket_shift"] = df["created_hour"].apply(hour_to_shift)
    
    breach_analysis = df.groupby(["ticket_shift", "channel"]).agg(
        total=("ticket_id", "count"),
        breaches=("sla_breach", "sum"),
        breach_rate=("sla_breach", "mean"),
    ).reset_index()
    
    # By hour
    hourly = df.groupby(["created_hour", "channel"]).agg(
        total=("ticket_id", "count"),
        breaches=("sla_breach", "sum"),
        breach_rate=("sla_breach", "mean"),
    ).reset_index()
    
    return breach_analysis, hourly


def compute_agent_leaderboard(tickets_df: pd.DataFrame, agents_df: pd.DataFrame) -> tuple:
    """Compute agent leaderboard.
    
    Policy §6: "Tier 2 agents are not to be compared with Tier 1 on volume metrics."
    Metric name: "Tickets closed per week"
    
    Returns:
        (tier1_df, tier2_df) - separate DataFrames.
    """
    from .preprocessing import resolve_agent_roster, compute_week_start
    
    df = tickets_df[
        (~tickets_df.get("is_duplicate", False)) &
        (tickets_df["status"].isin(config.CLOSURE_STATUSES))
    ].copy()
    
    # Enrich roster and week if not already present
    if "agent_team" not in df.columns or "agent_tier" not in df.columns:
        df = resolve_agent_roster(df, agents_df)
    if "week_start" not in df.columns:
        if "created_at" in df.columns:
            df = compute_week_start(df)
        else:
            df["week_start"] = "2026-01-01"
            
    # Tier 1 vs Tier 2 determination
    if "agent_tier" in df.columns and df["agent_tier"].notna().any():
        tier1 = df[df["agent_tier"] == 1].copy()
        tier2 = df[df["agent_tier"] == 2].copy()
    else:
        tier1_teams = {"Chat Frontline", "Email Frontline", "Voice Frontline"}
        tier1 = df[df["agent_team"].isin(tier1_teams)].copy()
        tier2 = df[df["agent_team"].str.contains("Escalations|Warranty", case=False, na=False)].copy()
    
    # Group by agent and week
    if len(tier1) > 0:
        agent_weekly = tier1.groupby(["agent_id", "week_start"]).agg(
            tickets_closed=("ticket_id", "count"),
            channel_mode=("channel", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
            median_handle_min=("handle_minutes", lambda x: x.median() if "handle_minutes" in tier1.columns and x.notna().any() else 0.0),
            auto_closed_share=("status", lambda x: (x == "closed").mean()),
            avg_csat=("csat_clean", lambda x: x.mean() if "csat_clean" in tier1.columns and x.notna().any() else np.nan),
            csat_n=("csat_clean", lambda x: x.notna().sum() if "csat_clean" in tier1.columns else 0),
        ).reset_index()
        
        # Aggregate across weeks
        agent_summary = agent_weekly.groupby("agent_id").agg(
            tickets_closed_per_week=("tickets_closed", "mean"),
            total_tickets=("tickets_closed", "sum"),
            weeks_active=("week_start", "nunique"),
            channel_mix=("channel_mode", lambda x: ", ".join(x.value_counts().head(3).index)),
            median_handle_min=("median_handle_min", "median"),
            auto_closed_share=("auto_closed_share", "mean"),
            avg_csat=("avg_csat", "mean"),
            csat_n=("csat_n", "sum"),
        ).reset_index()
        
        # Add agent info
        agent_info = agents_df[["agent_id", "team", "site", "shift"]].drop_duplicates()
        agent_summary = agent_summary.merge(agent_info, on="agent_id", how="left")
        
        # Sort by tickets closed per week
        agent_summary = agent_summary.sort_values("tickets_closed_per_week", ascending=False)
        agent_summary["rank"] = range(1, len(agent_summary) + 1)
        
        # Rename for display
        agent_summary.rename(columns={
            "tickets_closed_per_week": "Tickets closed per week",
        }, inplace=True)
    else:
        agent_summary = pd.DataFrame(columns=["agent_id", "Tickets closed per week", "rank"])
    
    # Add repeat rate if available
    repeat_path = f"{config.OUTPUTS_DIR}/repeat_contacts.csv"
    if os.path.exists(repeat_path) and len(agent_summary) > 0:
        try:
            repeats = pd.read_csv(repeat_path)
            if len(repeats) > 0 and "ticket_a_id" in repeats.columns:
                strict_repeats = repeats[repeats.get("is_strict", True) == True]
                repeat_counts = strict_repeats.groupby("ticket_a_id").size().reset_index(name="repeat_count")
                # Join back to tickets to get agent
                agent_repeats = df[df["ticket_id"].isin(repeat_counts["ticket_a_id"])].groupby("agent_id").size()
                agent_total = df.groupby("agent_id").size()
                agent_repeat_rate = (agent_repeats / agent_total * 100).fillna(0)
                agent_summary["repeat_rate_30d_pct"] = agent_summary["agent_id"].map(agent_repeat_rate).fillna(0)
        except Exception:
            pass
    
    # --- Tier 2 (unranked) ---
    if len(tier2) > 0:
        tier2_summary = tier2.groupby("agent_id").agg(
            total_tickets=("ticket_id", "count"),
            median_resolution_days=("handle_minutes", lambda x: (x / 1440).median() if "handle_minutes" in tier2.columns and x.notna().any() else 1.0),
            avg_csat=("csat_clean", lambda x: x.mean() if "csat_clean" in tier2.columns and x.notna().any() else np.nan),
            csat_n=("csat_clean", lambda x: x.notna().sum() if "csat_clean" in tier2.columns else 0),
        ).reset_index()
        agent_info = agents_df[["agent_id", "team", "site", "shift"]].drop_duplicates()
        tier2_summary = tier2_summary.merge(agent_info, on="agent_id", how="left")
    else:
        tier2_summary = pd.DataFrame(columns=["agent_id", "total_tickets", "median_resolution_days"])
    # No rank column for Tier 2
    
    # Save
    t1_path = f"{config.OUTPUTS_DIR}/agent_weekly_metrics.csv"
    t2_path = f"{config.OUTPUTS_DIR}/agent_tier2_metrics.csv"
    agent_summary.to_csv(t1_path, index=False)
    tier2_summary.to_csv(t2_path, index=False)
    
    return agent_summary, tier2_summary


def compute_category_product_cross(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of category x product."""
    df = tickets_df[~tickets_df["is_duplicate"]]
    cat_col = "classified_category" if "classified_category" in df.columns else "category"
    return pd.crosstab(df[cat_col], df["product_name"], margins=True)


def compute_lot_concentration(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Test whether complaint categories concentrate in specific lots/SKUs.
    
    Join orders.lot_code and test concentration beyond base rate.
    """
    df = tickets_df[
        (~tickets_df["is_duplicate"]) &
        (tickets_df["lot_code"].notna())
    ].copy()
    
    cat_col = "classified_category" if "classified_category" in df.columns else "category"
    
    # Categories to check for lot concentration
    hw_categories = ["Battery & Charging", "Connectivity & Bluetooth", 
                     "Physical Damage & Build", "Audio & Sound Quality",
                     "Charging & Battery", "Connectivity"]
    
    # Overall complaint rate by lot
    lot_total = df.groupby("lot_code").size().reset_index(name="total_complaints")
    
    # Complaint rate for specific categories by lot
    results = []
    for cat in hw_categories:
        cat_by_lot = df[df[cat_col] == cat].groupby("lot_code").size().reset_index(name="cat_complaints")
        merged = lot_total.merge(cat_by_lot, on="lot_code", how="left")
        merged["cat_complaints"] = merged["cat_complaints"].fillna(0)
        merged["cat_rate"] = merged["cat_complaints"] / merged["total_complaints"]
        
        # Overall rate for this category
        overall_rate = (df[cat_col] == cat).mean()
        
        # Flag lots with significantly higher rate (> 2x base rate and min 3 complaints)
        hot_lots = merged[
            (merged["cat_rate"] > overall_rate * 2) &
            (merged["cat_complaints"] >= 3)
        ].copy()
        
        if len(hot_lots) > 0:
            hot_lots["category"] = cat
            hot_lots["overall_rate"] = overall_rate
            hot_lots["concentration_ratio"] = merged["cat_rate"] / overall_rate
            results.append(hot_lots)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


import os

# Aliases
compute_agent_metrics = compute_agent_leaderboard
analyse_breaches_by_hour = compute_breach_by_hour_shift
