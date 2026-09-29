# DECISIONS.md — Vireo Support Intelligence

Each row: Decision | Evidence | Alternative Rejected

## Data Quality

| # | Decision | Evidence | Alternative Rejected |
|---|----------|----------|---------------------|
| 1 | Correct legacy `resolved_at` by +05:30 | 618/618 overlap tickets show exactly 5.50h offset; policy §9 says legacy event log stores UTC | Leave uncorrected (would give negative handle times for 60% of legacy tickets) |
| 2 | No currency correction on `refund_amount_inr` | Legacy/helpdesk median ratio = 0.95; both align with product retail prices | Apply 100x correction (no evidence of a scaling factor) |
| 3 | Keep `helpdesk` row for duplicates, flag `legacy_fd` as duplicate | All 653 dups are cross-source; helpdesk has correct timestamps | Keep legacy row (has wrong resolved_at); keep both (double-counts) |
| 4 | Convert CSAT 0 → NaN | Legacy uses 0 for no-response (2,083 cases), helpdesk uses NaN (4,856); policy §8 says exclude non-responses | Treat 0 as valid rating (would tank CSAT averages) |
| 5 | Exclude open/pending from closure metrics | Policy §10 defines attendance as resolved or closed; 644 tickets have null resolved_at and all are open/pending | Include them with imputed timestamps |
| 6 | Use Monday-start weeks in IST | Task spec says "Monday-start in IST"; policy timestamps are IST | Sunday-start or UTC weeks |

## Taxonomy

| # | Decision | Evidence | Alternative Rejected |
|---|----------|----------|---------------------|
| 7 | Use Gemma-discovered taxonomy of 12 categories | Intake-bot has 14.2% "Other"; Gemma identifies finer themes; human validation on test set | Use intake-bot categories directly (too noisy, large "Other") |
| 8 | Include `other` as a valid category | Some messages are truly unclassifiable or multi-issue | Force-classify everything (would reduce precision) |

## Architecture

| # | Decision | Evidence | Alternative Rejected |
|---|----------|----------|---------------------|
| 9 | Cache all LLM outputs to CSV/JSON files | Requirement: app must work with Ollama down | Re-run LLM on every app load |
| 10 | Use RRF with k=60 for hybrid retrieval | Standard parameter from literature; validated on labelled set | BM25 only (misses semantic similarity); vector only (misses exact matches) |
| 11 | Ollama models: `gemma4:latest` (chat), `nomic-embed-text:latest` (embeddings) | Output of `ollama list` | Hardcode model names without checking |
| 12 | Use `format` parameter for structured JSON output | Ollama supports JSON schema in format param; more reliable than prompt-only JSON | Regex extraction from free text (fragile) |

## Metrics

| # | Decision | Evidence | Alternative Rejected |
|---|----------|----------|---------------------|
| 13 | Right-censor last 30 days for repeat rate | Policy §10: 30-day window; tickets resolved after 2026-05-31 can't be judged | Include all tickets (inflates denominator, deflates rate) |
| 14 | Use channel-specific contact costs as primary | Policy §4 gives per-channel costs | Use blended ₹290 only (hides channel mix effects) |
| 15 | Report SLA breaches separately from missing first response | A missing response is a data issue, not necessarily a breach per policy §3 | Treat missing as breach (overstates breaches) |
| 16 | Agent leaderboard: Tier 1 only, with "not a performance score" note | Policy §6: Tier 2 measured on resolution days, not volume; task §6 requires separate view | Combine tiers (unfair comparison) |

## Known Limitations

- README.txt and email-thread.txt are empty (0 bytes) — cannot extract additional context from them
- No roster changes in data (all agents have open-ended to_date) — effective-date logic is simple but tested
- Lot code analysis limited by join rate between tickets and orders
