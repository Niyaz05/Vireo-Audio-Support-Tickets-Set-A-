# EDA Findings — Vireo Support Intelligence

Generated: 2026-09-29

## Data Shape

| Table     | Rows   | Columns | Notes                    |
|-----------|--------|---------|--------------------------|
| tickets   | 12,528 | 21      | Includes duplicates      |
| agents    | 44     | 8       | One row per assignment   |
| orders    | 15,000 | 8       | No nulls in key fields   |
| customers | 9,500  | 6       | Includes care_plus flag  |
| products  | 14     | 7       | 5 families               |

## 1. Timezones — CONFIRMED ✅

**Finding:** Legacy `resolved_at` is UTC; helpdesk `resolved_at` is IST. 653 duplicate tickets (same ticket_id in both sources) show an **exact 5.50-hour offset** between helpdesk and legacy `resolved_at` values.

- Legacy tickets with `resolved_at < created_at`: **2,263 out of 3,762** (60%)
- Legacy tickets with `resolved_at < first_response_at`: **2,472** (66%)
- Helpdesk: **zero** cases of resolved < created
- For the 653 overlap tickets: 618/618 (excl NaT) show exactly +5:30 offset

**Decision:** Correct legacy `resolved_at` by adding +05:30. Keep raw value in `resolved_at_raw`.

## 2. Legacy Currency — NOT CONFIRMED ❌

**Finding:** Refund amounts are in the same scale across sources. No 100x factor.

- Legacy refund median: ₹2,374 | Helpdesk median: ₹2,499
- Legacy/helpdesk ratio: 0.95 (median), 1.09 (mean)
- Refund/retail price ratio: legacy 0.77 ± 0.41, helpdesk 0.76 ± 0.42

Both distributions match product retail prices. The "native unit" mentioned in policy section 9 does not affect refund_amount_inr in this dataset — amounts are already in INR.

**Decision:** No currency correction needed. Document in Data Quality page.

## 3. Duplicates — CONFIRMED ✅

**Finding:** 653 ticket_ids appear exactly twice, always one `helpdesk` + one `legacy_fd`. All are cross-source duplicates (no same-source dups). Content-key dedup confirms all 653 are true duplicates.

- All duplicate tickets have `created_at` before 2025-09-14 (the cutover)
- Total rows: 12,528 → unique tickets: 11,875 after dedup

**Decision:** Keep the `helpdesk` row (correct timestamps), flag the `legacy_fd` duplicate with `dq_flags = 'duplicate_legacy'`. Exclude duplicates from all headline metrics.

## 4. CSAT — CONFIRMED ✅

**Finding:** Legacy system records "no response" as `0`. Helpdesk records "no response" as `NaN`.

- Legacy csat=0: **2,083** (55% of legacy tickets)
- Legacy csat 1-5: 1,679 (45%)
- Helpdesk csat=NaN: **4,856** (55% of helpdesk)
- Helpdesk csat 1-5: 3,910 (45%)

Closed (auto-closed) tickets: 1,168 total, 510 with valid CSAT (1-5), 198 with csat=0, 460 with NaN. Policy §8 says auto-closed tickets are surveyed.

**Decision:** Convert 0 to NaN in both sources. Never average zeros. Exclude NaN from CSAT calculations.

## 5. Status Distribution

| Status   | Count  | Treatment              |
|----------|--------|------------------------|
| resolved | 10,716 | Include in closure      |
| closed   | 1,168  | Include (auto-closed)   |
| open     | 399    | Exclude from closure    |
| pending  | 245    | Exclude from closure    |

Open/pending tickets have no `resolved_at` (644 null resolved_at = 399 open + 245 pending).

## 6. Order Joins

- Tickets with order_id: **8,310** (66%)
- Tickets without order_id: **4,218** (34%)
- All 5,063 unique order_ids in tickets match orders.csv
- Missing order_ids can use customer_id + product_sku + date fallback

## 7. Agent Roster

- All 44 agents in tickets appear in the roster
- **Zero** overlapping roster rows
- **Zero** agents missing from roster
- Tier 1: 38 agents, Tier 2: 6 agents (Escalations & Warranty)
- All roster rows have open-ended `to_date` (no role changes in data)

## 8. Refund + Replacement Conflicts

**4 tickets** have both refund > 0 AND replacement_issued = Y (violates policy §5):

| ticket_id  | order_id | refund   | reason_code  | source    |
|------------|----------|----------|--------------|-----------|
| TK-241405  | VR892778 | ₹2,499  | RETURN-QC-OK | legacy_fd |
| TK-244372  | VR908861 | ₹2,974  | GW-OTHER     | legacy_fd |
| TK-250483  | VR883328 | ₹2,878  | DOA-REPL     | helpdesk  |
| TK-252411  | VR885453 | ₹3,499  | RETURN-QC-OK | helpdesk  |

## 9. Refund Reason Code Issues

- **GW-OTHER** used 44 times: mostly on hardware categories (Connectivity 14, Charging & Battery 11, Warranty & Repair 10, Audio 8). Appears to be a catch-all.
- **DOA-REPL** used 144 times: 143/144 have replacement_issued=N (refund chosen). Spread across Delivery (64), Warranty (30), Audio (15), Battery (12), Connectivity (10), App/Firmware (7).

## 10. Transfers

| Transfers | Count  |
|-----------|--------|
| 0         | 11,424 |
| 1         | 967    |
| 2         | 137    |

Legacy system does have transfer counts (not mentioned in policy as helpdesk-only — it exists in both sources).

## 11. Language & Text

- **IVR transcripts**: 1,321 messages contain `[IVR transcript]` markers with transcription noise (e.g., "kiked" for "kicked")
- **Hinglish**: ~316 messages contain Hindi/Hinglish markers
- Messages are generally short, informal, often multi-line with greetings

## 12. Date Edges

- Data window: 2025-01-01 to 2026-06-30 (545 days, ~78 full weeks)
- Partial first week: 2024-12-30 (Mon) — 59 tickets (Wed-only)
- Partial last week: 2026-06-29 (Mon) — 46 tickets (Mon-Tue only)
- Right-censoring cutoff for repeat contacts: 2026-05-31

## 13. Category Distribution (intake-bot)

| Category              | Count | Share  |
|-----------------------|-------|--------|
| Delivery & Shipping   | 2,260 | 18.0%  |
| Other                 | 1,780 | 14.2%  |
| Billing & Payments    | 1,717 | 13.7%  |
| Returns & Refunds     | 1,264 | 10.1%  |
| Connectivity          | 1,200 | 9.6%   |
| Charging & Battery    | 1,002 | 8.0%   |
| App & Firmware        | 861   | 6.9%   |
| Audio Quality         | 825   | 6.6%   |
| Warranty & Repair     | 665   | 5.3%   |
| Product Enquiry       | 625   | 5.0%   |
| Account & Login       | 329   | 2.6%   |

Note: 11 categories from intake bot. "Other" is large (14.2%), suggesting significant miscategorisation.

## 14. Product Families

| Family     | Products | Price Range       |
|------------|----------|-------------------|
| earbuds    | 4        | ₹1,299 – ₹3,499 |
| headphones | 2        | ₹4,999 – ₹6,999 |
| speaker    | 2        | ₹2,499 – ₹5,999 |
| watch      | 3        | ₹1,999 – ₹6,499 |
| accessory  | 3        | ₹399 – ₹1,499   |
