#!/usr/bin/env python3
"""Comprehensive EDA script for Vireo Support Intelligence.
Investigates all data traps from the task spec."""

import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 80)

# Load data
tickets = pd.read_csv('tickets.csv')
agents = pd.read_csv('agents.csv')
orders = pd.read_csv('orders.csv')
customers = pd.read_csv('customers.csv')
products = pd.read_csv('products.csv')

print("=" * 80)
print("SECTION 1: BASIC SHAPE")
print("=" * 80)
print(f"tickets:   {tickets.shape[0]:>6} rows x {tickets.shape[1]} cols")
print(f"agents:    {agents.shape[0]:>6} rows x {agents.shape[1]} cols")
print(f"orders:    {orders.shape[0]:>6} rows x {orders.shape[1]} cols")
print(f"customers: {customers.shape[0]:>6} rows x {customers.shape[1]} cols")
print(f"products:  {products.shape[0]:>6} rows x {products.shape[1]} cols")

print("\n--- Tickets dtypes & nulls ---")
print(tickets.dtypes)
print(tickets.isnull().sum())

print("\n--- source_system values ---")
print(tickets['source_system'].value_counts())

print("\n--- status values ---")
print(tickets['status'].value_counts())

print("\n--- channel values ---")
print(tickets['channel'].value_counts())

print("\n--- category values ---")
print(tickets['category'].value_counts())

print("\n--- priority values ---")
print(tickets['priority'].value_counts())

print("\n--- assigned_team values ---")
print(tickets['assigned_team'].value_counts())

print("\n--- replacement_issued values ---")
print(tickets['replacement_issued'].value_counts())

# Parse timestamps
print("\n" + "=" * 80)
print("SECTION 2: TIMESTAMPS & TIMEZONE TRAP")
print("=" * 80)

tickets['created_at'] = pd.to_datetime(tickets['created_at'])
tickets['first_response_at'] = pd.to_datetime(tickets['first_response_at'])
tickets['resolved_at'] = pd.to_datetime(tickets['resolved_at'])

print("\n--- Date range ---")
print(f"created_at:        {tickets['created_at'].min()} to {tickets['created_at'].max()}")
print(f"first_response_at: {tickets['first_response_at'].min()} to {tickets['first_response_at'].max()}")
print(f"resolved_at:       {tickets['resolved_at'].min()} to {tickets['resolved_at'].max()}")

# Compute handle time and check for negatives
tickets['resolution_time_min'] = (tickets['resolved_at'] - tickets['created_at']).dt.total_seconds() / 60
tickets['fr_time_min'] = (tickets['first_response_at'] - tickets['created_at']).dt.total_seconds() / 60

legacy = tickets[tickets['source_system'] == 'legacy_fd']
helpdesk = tickets[tickets['source_system'] == 'helpdesk']

print(f"\n--- Legacy tickets: {len(legacy)} ---")
print(f"  resolved_at < created_at: {(legacy['resolved_at'] < legacy['created_at']).sum()}")
print(f"  resolved_at < first_response_at: {(legacy['resolved_at'] < legacy['first_response_at']).sum()}")
print(f"  resolution_time_min stats:")
print(legacy['resolution_time_min'].describe())

print(f"\n--- Helpdesk tickets: {len(helpdesk)} ---")
print(f"  resolved_at < created_at: {(helpdesk['resolved_at'] < helpdesk['created_at']).sum()}")
print(f"  resolved_at < first_response_at: {(helpdesk['resolved_at'] < helpdesk['first_response_at']).sum()}")
print(f"  resolution_time_min stats:")
print(helpdesk['resolution_time_min'].describe())

# Test for ~5.5h offset
# For tickets that appear in both sources (same ticket_id), compare resolved_at
both_ids = set(legacy['ticket_id']) & set(helpdesk['ticket_id'])
print(f"\n--- Tickets appearing in BOTH sources: {len(both_ids)} ---")
if both_ids:
    sample_ids = list(both_ids)[:10]
    for tid in sample_ids:
        l = legacy[legacy['ticket_id'] == tid].iloc[0]
        h = helpdesk[helpdesk['ticket_id'] == tid].iloc[0]
        diff = (h['resolved_at'] - l['resolved_at']).total_seconds() / 3600
        print(f"  {tid}: helpdesk={h['resolved_at']}, legacy={l['resolved_at']}, diff={diff:.2f}h")

# Check offset distribution for legacy vs helpdesk
# For the same ticket appearing in both, the difference should reveal the timezone offset
if both_ids:
    diffs = []
    for tid in both_ids:
        l = legacy[legacy['ticket_id'] == tid].iloc[0]
        h = helpdesk[helpdesk['ticket_id'] == tid].iloc[0]
        diff = (h['resolved_at'] - l['resolved_at']).total_seconds() / 3600
        diffs.append(diff)
    diffs = np.array(diffs)
    print(f"\n  Offset distribution (helpdesk - legacy resolved_at) in hours:")
    print(f"    mean={diffs.mean():.2f}, median={np.median(diffs):.2f}, std={diffs.std():.2f}")
    print(f"    min={diffs.min():.2f}, max={diffs.max():.2f}")
    print(f"    Exactly 5.5h: {np.sum(np.isclose(diffs, 5.5, atol=0.01))}")

print("\n" + "=" * 80)
print("SECTION 3: DUPLICATES / RE-IMPORTS")
print("=" * 80)

# ticket_id duplicates
dup_ids = tickets['ticket_id'].value_counts()
dup_ids = dup_ids[dup_ids > 1]
print(f"Duplicate ticket_ids: {len(dup_ids)} ids with >1 row ({dup_ids.sum()} total rows)")
if len(dup_ids) > 0:
    print(f"  Counts: {dict(dup_ids.value_counts())}")
    # Check if duplicates cross source_system
    dup_tickets = tickets[tickets['ticket_id'].isin(dup_ids.index)]
    cross = dup_tickets.groupby('ticket_id')['source_system'].nunique()
    print(f"  Cross-source duplicates: {(cross > 1).sum()}")
    print(f"  Same-source duplicates: {(cross == 1).sum()}")

# Content-based duplicates
tickets['msg_norm'] = tickets['customer_message'].str.lower().str.strip()
content_key = tickets.groupby(['customer_id', 'created_at', 'channel', 'product_sku', 'msg_norm']).size()
content_dups = content_key[content_key > 1]
print(f"Content-key duplicates: {len(content_dups)} groups")

print("\n" + "=" * 80)
print("SECTION 4: LEGACY CURRENCY TRAP")
print("=" * 80)

tickets['refund_amount_inr'] = pd.to_numeric(tickets['refund_amount_inr'], errors='coerce')

has_refund = tickets[tickets['refund_amount_inr'] > 0]
print(f"Tickets with refund > 0: {len(has_refund)}")

legacy_refunds = has_refund[has_refund['source_system'] == 'legacy_fd']
helpdesk_refunds = has_refund[has_refund['source_system'] == 'helpdesk']

print(f"\n--- Legacy refund_amount_inr ({len(legacy_refunds)} tickets) ---")
if len(legacy_refunds) > 0:
    print(legacy_refunds['refund_amount_inr'].describe())
    print(f"  Some values: {legacy_refunds['refund_amount_inr'].head(10).tolist()}")

print(f"\n--- Helpdesk refund_amount_inr ({len(helpdesk_refunds)} tickets) ---")
if len(helpdesk_refunds) > 0:
    print(helpdesk_refunds['refund_amount_inr'].describe())
    print(f"  Some values: {helpdesk_refunds['refund_amount_inr'].head(10).tolist()}")

# Compare with retail prices
print(f"\n--- Product retail prices ---")
print(products[['sku', 'retail_price_inr', 'unit_cost_inr']].to_string())

# Check ratio
if len(legacy_refunds) > 0 and len(helpdesk_refunds) > 0:
    print(f"\n  Legacy median / Helpdesk median: {legacy_refunds['refund_amount_inr'].median() / helpdesk_refunds['refund_amount_inr'].median():.2f}")
    print(f"  Legacy mean / Helpdesk mean: {legacy_refunds['refund_amount_inr'].mean() / helpdesk_refunds['refund_amount_inr'].mean():.2f}")

# Check legacy refunds against products
if len(legacy_refunds) > 0:
    merged = legacy_refunds.merge(products, left_on='product_sku', right_on='sku', how='left')
    refund_col = 'refund_amount_inr_x' if 'refund_amount_inr_x' in merged.columns else 'refund_amount_inr'
    merged['ratio_to_retail'] = merged[refund_col] / merged['retail_price_inr']
    print(f"\n  Legacy refund / retail price ratio:")
    print(merged['ratio_to_retail'].describe())

print("\n" + "=" * 80)
print("SECTION 5: CSAT TRAP")
print("=" * 80)

print(f"\n--- csat_score value distribution ---")
tickets['csat_raw'] = pd.to_numeric(tickets['csat_score'], errors='coerce')
print(tickets['csat_raw'].value_counts(dropna=False).sort_index())

# Legacy zeros
legacy_csat = legacy['csat_score'].copy()
print(f"\n  Legacy csat=0: {(legacy_csat == '0').sum()}")
print(f"  Legacy csat blank: {legacy_csat.isna().sum() + (legacy_csat == '').sum()}")

# Helpdesk zeros
helpdesk_csat = helpdesk['csat_score'].copy()
print(f"  Helpdesk csat=0: {(helpdesk_csat == '0').sum()}")
print(f"  Helpdesk csat blank: {helpdesk_csat.isna().sum() + (helpdesk_csat == '').sum()}")

# CSAT on auto-closed
# Check if status=closed exists and if it has CSAT
print(f"\n  Status=closed tickets: {(tickets['status'] == 'closed').sum()}")
closed_tickets = tickets[tickets['status'] == 'closed']
print(f"  Closed with CSAT (1-5): {closed_tickets['csat_raw'].between(1, 5).sum()}")
print(f"  Closed with CSAT=0: {(closed_tickets['csat_raw'] == 0).sum()}")
print(f"  Closed with blank CSAT: {closed_tickets['csat_raw'].isna().sum()}")

print("\n" + "=" * 80)
print("SECTION 6: ORDER JOIN RATES")
print("=" * 80)

print(f"  Tickets with order_id: {tickets['order_id'].notna().sum() - (tickets['order_id'] == '').sum()}")
print(f"  Tickets without order_id: {tickets['order_id'].isna().sum() + (tickets['order_id'] == '').sum()}")

# Fallback join on customer_id + product_sku
no_order = tickets[(tickets['order_id'].isna()) | (tickets['order_id'] == '')]
print(f"\n  Tickets without order_id: {len(no_order)}")
print(f"  Of those with product_sku: {no_order['product_sku'].notna().sum() - (no_order['product_sku'] == '').sum()}")

# Check if order_id in tickets matches orders
valid_order_ids = set(orders['order_id'])
ticket_order_ids = set(tickets[tickets['order_id'].notna() & (tickets['order_id'] != '')]['order_id'])
print(f"\n  Unique order_ids in tickets: {len(ticket_order_ids)}")
print(f"  Of those found in orders.csv: {len(ticket_order_ids & valid_order_ids)}")
print(f"  Missing from orders.csv: {len(ticket_order_ids - valid_order_ids)}")

print("\n" + "=" * 80)
print("SECTION 7: AGENT / ROSTER")
print("=" * 80)

print(f"  Unique agents in roster: {agents['agent_id'].nunique()}")
print(f"  Unique agents in tickets: {tickets['agent_id'].nunique()}")

ticket_agents = set(tickets['agent_id'].dropna())
roster_agents = set(agents['agent_id'])
print(f"  Agents in tickets but not in roster: {ticket_agents - roster_agents}")
print(f"  Agents in roster but not in tickets: {roster_agents - ticket_agents}")

# Overlapping roster rows
agents['from_date'] = pd.to_datetime(agents['from_date'])
agents['to_date'] = pd.to_datetime(agents['to_date'])
overlap_count = 0
for aid in agents['agent_id'].unique():
    rows = agents[agents['agent_id'] == aid].sort_values('from_date')
    if len(rows) > 1:
        for i in range(len(rows) - 1):
            r1 = rows.iloc[i]
            r2 = rows.iloc[i + 1]
            if pd.isna(r1['to_date']) or r1['to_date'] >= r2['from_date']:
                overlap_count += 1
print(f"  Overlapping roster rows: {overlap_count}")

# Team distribution
print(f"\n--- Agent teams ---")
print(agents['team'].value_counts())
print(f"\n--- Agent tiers ---")
print(agents['tier'].value_counts())

print("\n" + "=" * 80)
print("SECTION 8: REFUND + REPLACEMENT CONFLICT")
print("=" * 80)

refund_and_replace = tickets[(tickets['refund_amount_inr'] > 0) & (tickets['replacement_issued'] == 'Y')]
print(f"  Tickets with BOTH refund > 0 AND replacement=Y: {len(refund_and_replace)}")
if len(refund_and_replace) > 0:
    print(refund_and_replace[['ticket_id', 'order_id', 'refund_amount_inr', 'replacement_issued', 'refund_reason_code', 'source_system']].head(20).to_string())

# Check refund_reason_code distribution
print(f"\n--- refund_reason_code values ---")
print(tickets['refund_reason_code'].value_counts(dropna=False))

# GW-OTHER usage
gw_other = tickets[tickets['refund_reason_code'] == 'GW-OTHER']
print(f"\n  GW-OTHER tickets: {len(gw_other)}")
if len(gw_other) > 0:
    print(f"  GW-OTHER categories: {dict(gw_other['category'].value_counts())}")

# DOA-REPL on non-DOA tickets
doa_repl = tickets[tickets['refund_reason_code'] == 'DOA-REPL']
print(f"\n  DOA-REPL tickets: {len(doa_repl)}")
if len(doa_repl) > 0:
    print(f"  DOA-REPL categories: {dict(doa_repl['category'].value_counts())}")
    print(f"  DOA-REPL replacement_issued: {dict(doa_repl['replacement_issued'].value_counts())}")

print("\n" + "=" * 80)
print("SECTION 9: LOT CODES")
print("=" * 80)

print(f"  Unique lot codes in orders: {orders['lot_code'].nunique()}")
print(f"  Orders with lot_code: {orders['lot_code'].notna().sum()}")

# We need to join tickets to orders to check lot concentration
# For now just show lot code distribution
print(f"\n  Top lot codes:")
print(orders['lot_code'].value_counts().head(20))

print("\n" + "=" * 80)
print("SECTION 10: LANGUAGE / TEXT")
print("=" * 80)

# Sample messages for Hinglish/transliteration
msgs = tickets['customer_message'].dropna().head(30)
for i, m in enumerate(msgs[:10]):
    print(f"  [{i}] {m[:120]}")

# Check for IVR markers
ivr_markers = tickets['customer_message'].str.contains(r'(?i)(ivr|transcript|voicemail|callback)', na=False)
print(f"\n  Messages with IVR-like markers: {ivr_markers.sum()}")

# Check for Hindi/Hinglish
hindi_markers = tickets['customer_message'].str.contains(r'(?i)(kya|hai|nahi|mera|karo|bhai|yaar|please\s+kar|abhi|batao)', na=False)
print(f"  Messages with Hinglish markers: {hindi_markers.sum()}")

print("\n" + "=" * 80)
print("SECTION 11: DATE EDGES")
print("=" * 80)

print(f"  Data start: {tickets['created_at'].min()}")
print(f"  Data end:   {tickets['created_at'].max()}")
print(f"  Date range: {(tickets['created_at'].max() - tickets['created_at'].min()).days} days")

# Weekly volume
tickets['week_start'] = tickets['created_at'].dt.to_period('W-SUN').apply(lambda x: x.start_time)
weekly_vol = tickets.groupby('week_start').size()
print(f"\n  First week: {weekly_vol.index[0]} - {weekly_vol.iloc[0]} tickets")
print(f"  Last week:  {weekly_vol.index[-1]} - {weekly_vol.iloc[-1]} tickets")

# Right-censoring for repeat contacts
data_end = tickets['created_at'].max()
print(f"\n  Data end date: {data_end}")
print(f"  30-day cutoff for repeat censoring: {data_end - pd.Timedelta(days=30)}")

# Tickets per week
print(f"\n--- Weekly volume (first/last 5 weeks) ---")
print(weekly_vol.head())
print("...")
print(weekly_vol.tail())

print("\n" + "=" * 80)
print("SECTION 12: TRANSFERS")
print("=" * 80)

tickets['transfers'] = pd.to_numeric(tickets['transfers'], errors='coerce')
print(tickets['transfers'].value_counts(dropna=False).sort_index())
print(f"\n  Legacy transfers: {legacy['transfers'].value_counts(dropna=False).to_dict()}")

print("\n" + "=" * 80)
print("SECTION 13: MISSING VALUES SUMMARY")
print("=" * 80)
for col in tickets.columns:
    n_missing = tickets[col].isna().sum()
    n_empty = (tickets[col] == '').sum() if tickets[col].dtype == 'object' else 0
    if n_missing > 0 or n_empty > 0:
        print(f"  {col}: {n_missing} null, {n_empty} empty")

print("\n\nEDA COMPLETE")
