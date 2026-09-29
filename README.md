# Vireo Support Intelligence

A local, reproducible analytics app for Vireo Audio's customer-support tickets (1 Jan 2025 – 30 Jun 2026).
It turns raw tickets into a weekly view of what customers complain about, what it costs, and where money can be saved.

**Design rule: Python calculates every number. Gemma 4 (via Ollama) only reads and writes language.
A human-labelled validation set tells you how far to trust Gemma.**

All inference is local through Ollama. No cloud APIs, no paid calls, no per-ticket model bill.

---

## 1. Headline results (current run)

| Item | Result |
|---|---|
| Support tickets in the export | 12,528 |
| Duplicate (re-imported) records flagged | 653 |
| Unique tickets after de-duplication | 11,875 |
| Human-labelled validation tickets | 50 labelled, **49 evaluated** (1 had no cached classification) |
| Classifier accuracy on those 49 | **69.4%** (34 of 49); 95% interval roughly 54% – 82% |
| Macro-F1 on those 49 | **59.5%** |
| Intake-bot baseline on the same 49 | 32.7% accuracy, 27.9% macro-F1 |
| Top business opportunity | Reduce DOA-driven replacements |
| Scenario saving | about **Rs 137,139 per quarter** (a scenario, not realised savings) |

Please read these correctly:

- The 69.4% and 59.5% describe **the 49 evaluated tickets only**. They are not the accuracy of all 11,875 tickets, and with n = 49 the uncertainty is wide.
- The Rs 137,139 rests on an **assumed** 30% reduction in DOA replacements (1.1% of tickets to 0.8%). It is a scenario. Nothing here proves a saving without a deployed change.

---

## 2. Quick start

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Ollama (keep running in a separate terminal)
ollama serve

# 3. Check the exact model names on your machine, then pull if missing
ollama list
ollama pull gemma4:latest
ollama pull nomic-embed-text:latest

# 4. Smoke test
python -m src.pipeline --limit 30 --skip-embed

# 5. Full pipeline
python -m src.pipeline

# 6. Human validation: label outputs/labelling_sheet_50.csv, save as outputs/human_labels.csv
#    (see section 5), then re-run evaluation:
python -m src.pipeline --skip-embed --skip-classify

# 7. Dashboard
streamlit run app.py
```

The repository already contains cached outputs (classifications, metrics, digests, human labels), so
`streamlit run app.py` works immediately after step 1, without Ollama running. Ollama is only needed to
re-classify tickets, write new digests, or use "Ask the Tickets".

**Runtime.** The last recorded classification run (`outputs/run_stats.json`) took about 39 minutes. A full
run needing Gemma for many tickets takes roughly 40–50 minutes on the author's Apple-silicon Mac. Runtime
depends heavily on your hardware and on how much of the cache is reused. Classification is resumable.

---

## 3. Requirements

| Requirement | Details |
|---|---|
| Python | 3.9+ (developed on 3.9.6) |
| Ollama | https://ollama.com, with `ollama serve` running |
| Models | `gemma4:latest` (chat), `nomic-embed-text:latest` (embeddings) |
| RAM | 8 GB minimum for Gemma 4 |
| Packages | pandas, numpy, streamlit, rank_bm25, requests, pyarrow, pytest (`requirements.txt`, unpinned) |

### Configuration

Model names are never hard-coded in logic. Defaults are in `src/config.py`; override with environment variables:

```bash
export OLLAMA_MODEL="gemma4:latest"
export OLLAMA_EMBED_MODEL="nomic-embed-text:latest"
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Data

Copy the pack files into `data/` as **real files, not symlinks**:

```
data/tickets.csv   data/agents.csv   data/customers.csv
data/orders.csv    data/products.csv
data/support-policy.pdf  data/README.txt  data/email-thread.txt   (reference only, not loaded by code)
```

The client data files are not committed to this repository because they contain customer information.

---

## 4. Running the pipeline

```bash
python -m src.pipeline                   # full run
python -m src.pipeline --limit 200       # first N tickets, smoke test
python -m src.pipeline --skip-embed      # skip embedding generation
python -m src.pipeline --skip-classify   # reuse the existing classification cache
```

The pipeline runs 8 steps in order:

1. **Preprocess.** Clean timestamps, flag duplicates, join orders and products, compute SLA and costs, mask PII.
2. **Human validation setup.** Create the 50-ticket labelling sheet for independent evaluation.
3. **Hybrid retrieval index.** Build the BM25 index and optionally generate vector embeddings through Ollama.
4. **Classification.** Exact-cache reuse, BM25 similarity reuse, then Gemma 4 batch classification (up to 8 tickets per request) with structured-output validation and resumable caching.
5. **Repeat contacts.** Rule-based 30-day matching (strict and loose) plus the "colleague" claim test.
6. **Weekly analytics.** Volume, category share, SLA, costs, CSAT, Tier 1 agent throughput, breach profile.
7. **Opportunity engine.** Rank candidate savings as Observed, Estimated or Scenario.
8. **Human validation and digests.** Compare predictions with the completed human labels and generate weekly executive digests.

Results are cached in `outputs/classified_tickets.csv`, so already-classified tickets are reused on later runs.

### How classification works (and its trade-off)

```
Unique messages
   │
   ▼
Exact message in cache? ── yes ──► reuse result                (status ok / cached)
   │ no
   ▼
Very similar message already classified? (BM25 score ≥ 25) ── yes ──► copy its category   (status sim_reuse)
   │ no
   ▼
Gemma 4 via Ollama, batch of 8, JSON-schema output, 12 fixed categories   (status ok)
   │
   ▼
Validate JSON ─► save to cache (resumable)
```

Similarity reuse avoids a Gemma call, but the copied label was **not produced by Gemma for that ticket**.
In the current cache 6,817 of 11,566 rows (59%) are `sim_reuse` and 4,749 (41%) were classified by Gemma.
On the 49 evaluated tickets, Gemma-classified rows scored 78.9% (15 of 19) and reused rows 63.3% (19 of 30).
The samples are tiny, but the direction suggests the shortcut costs accuracy. The threshold of 25 was not tuned.

---

## 5. Human validation: label 50 tickets

The classifier is evaluated on a 50-ticket, human-labelled sample. The application never generates its own labels.

After the first pipeline run, `outputs/labelling_sheet_50.csv` is created. It holds 50 unique, usable tickets selected as a stratified sample.

1. Open `outputs/labelling_sheet_50.csv` and read each `customer_message`.
2. Fill `human_label` with exactly one of the 12 categories:

   ```
   Audio & Sound Quality      Battery & Charging
   Connectivity & Bluetooth   Delivery & Shipping
   App & Firmware Issues      Returns & Refunds
   Billing & Payment          Warranty & Repair
   Account & Login            Product Enquiry
   Physical Damage & Build    Other
   ```

3. Label independently of Gemma's prediction. The human label is the ground truth.
4. Save the completed file as `outputs/human_labels.csv`.
5. Re-run using the cached classifications:

   ```bash
   python -m src.pipeline --skip-embed --skip-classify
   ```

If `human_labels.csv` is missing or empty, the code raises an error and skips evaluation. It does not invent labels.

The evaluation writes `outputs/evaluation_metrics.json` and the **Validation & Evaluation** page shows:
accuracy, error rate, macro-F1, per-category precision/recall/F1, confusion matrix, misclassified examples,
confidence calibration, and comparison with the intake-bot baseline.

### Current validation result

| Metric | Result |
|---|---|
| Human-labelled tickets | 50 (49 evaluated) |
| Accuracy | 69.4% |
| Error rate | 30.6% |
| Macro-F1 | 59.5% |
| Intake-bot baseline accuracy | 32.7% |

What the results show:

- The categories with the most trouble are `Other` (0 of 2 correct), `Physical Damage & Build` (0 of 2, confused with `Warranty & Repair`), `Product Enquiry` (recall 33%) and `Returns & Refunds` (recall 58%, the largest class with 12 tickets).
- Confidence is not informative. 46 of 49 tickets fall in the 0.85–1.00 bucket, where accuracy is 69.6%, about the same as overall. Treat confidence as a heuristic, not a probability.
- Per-category figures rest on 1 to 12 tickets each and are anecdotal.
- Labels come from a single labeller, with no second-person check.

### Validation workflow

```
12,528 tickets ─► Preprocessing ─► 11,875 unique ─► Gemma 4 classifier ─► classification cache
                                                                              │
                                    ┌─────────────────────────────────────────┤
                                    ▼                                         ▼
                           automated pipeline                        human validation
                        (analytics, digests)                   50 tickets ─► manual labels
                                    │                                         │
                                    └───────────────────┬─────────────────────┘
                                                        ▼
                                                   Evaluation
                                              ┌─────────┴─────────┐
                                           Accuracy 69.4%     Macro-F1 59.5%
                                              (49 evaluated tickets)
```

---

## 6. Dashboard pages

| Page | What to look at |
|---|---|
| Overview | KPI cards and the top business opportunity |
| Weekly Digest | Pick a week: metrics, written digest, masked example complaints |
| Complaint Analysis | Category distribution, trend, category by product and channel, filters |
| Ask the Tickets | Hybrid search (BM25 + vectors + RRF) returns the 10 closest tickets; Gemma answers from those only. Needs Ollama running |
| Agent Throughput | "Tickets closed per week" for Tier 1 agents. Tier 2 is separate and unranked. This is not a performance score |
| Repeat Contacts | Strict vs loose repeat rate, breakdowns, the colleague-claim test |
| Opportunity Engine | Candidate savings labelled Observed / Estimated / Scenario, with evidence grade |
| Validation & Evaluation | Metrics from section 5 |
| Data Quality | Duplicates, timestamp fixes, CSAT issues, join rates, roster and refund checks |
| Run Stats | Last classification run and the zero-marginal-cost note |

### Reading the Weekly Digest

- **Source:** `gemma_verified` means Gemma wrote it. `fallback_*` means a deterministic template wrote it because Gemma was offline or failed.
- **Guardrail:** extracts every number in the digest and checks it against the calculated facts. Template digests pass by construction, so "Guardrail Verified" on a fallback digest only means the numbers match the facts.
- **Facts hash:** identifies the exact facts used, for caching.
- The final week (starting 2026-06-29) is partial, and the last 30 days are right-censored for repeat contacts, so low values there are expected.

---

## 7. What is implemented

| Area | What it does | File |
|---|---|---|
| Data loading | Loads tickets, agents, orders, customers, products | `src/data_loader.py` |
| Cleaning and enrichment | Timezone fix, duplicate flagging, CSAT cleaning, joins, SLA, costs, PII masking | `src/preprocessing.py` |
| Hybrid retrieval | BM25 + Ollama embeddings merged with Reciprocal Rank Fusion (k = 60) | `src/retrieval.py` |
| Classification | Exact cache, BM25 similarity reuse and Gemma 4 batch classification (8 per request), resumable cache | `src/classifier.py` |
| Repeat contacts | Rule-based 30-day matching, strict and loose, colleague-claim test | `src/repeat_contacts.py` |
| Weekly analytics | Volume, category share, week-over-week change, SLA, transfers, costs, CSAT, agent metrics | `src/analytics.py` |
| Opportunity engine | Ranks candidate savings as Observed / Estimated / Scenario | `src/opportunity.py` |
| Validation | 50-ticket human labelling sheet, accuracy, error rate, macro-F1, per-category metrics, confusion matrix, confidence calibration, digest guardrail | `src/validation.py` |
| Weekly digest | Gemma writes a six-heading digest from a facts JSON, with a number-checking guardrail | `src/digest.py` |
| Pipeline | Runs everything end to end | `src/pipeline.py` |
| Dashboard | Streamlit app, 10 pages | `app.py` |
| Tests | 14 pytest tests for deterministic logic (all pass) | `tests/test_vireo.py` |

### Business rules (from `support-policy.pdf`)

| Rule | Value | Policy |
|---|---|---|
| First-response SLA | chat 15 min, voice 2 h, social 4 h, email 8 h | §3 |
| SLA breach credit | Rs 350 per breached ticket | §3 |
| Contact cost | chat 210, email 260, voice 520, social 240 (blended 290) | §4 |
| Transfer cost | Rs 305 per transfer | §4 |
| Agent cost | Rs 165 per agent-hour, 8-hour shifts | §4 |
| Replacement cost | product unit cost + Rs 340 | §5 |
| Goodwill cap | Rs 500 per ticket | §5 |
| CSAT | 1–5; blank and legacy 0 are "no response", excluded from averages | §8 |
| Shifts | Morning 06–14, Day 14–22, Night 22–06 (IST) | §7 |
| Repeat contact | same customer, same issue, within 30 days of resolution | §10 |
| Helpdesk cutover | 14 September 2025 | §9 |
| Tier 2 (Warranty) | never ranked against Tier 1 on volume | §6 |

### Data-quality decisions

- **Timezone:** legacy `resolved_at` is UTC and helpdesk is IST. 618 of 618 overlapping tickets showed an exact 5.5-hour offset, so legacy values are shifted +5:30 and the raw value is kept in `resolved_at_raw`.
- **Duplicates:** 653 legacy tickets were re-imported. The helpdesk row is kept and the legacy row is flagged `is_duplicate`. Nothing is deleted.
- **CSAT:** 0 and blank become missing.
- **Closure:** only `resolved` and `closed` count as closed.
- **Repeat rate:** the last 30 days are right-censored.
- **Currency:** no correction applied to legacy refunds, because the evidence showed no scaling factor.
- **Colleague claim:** transferred tickets repeat slightly less (3.8%) than non-transferred (4.6%), so the data does not support the "I already told your colleague" pattern.

---

## 8. Output files

| File | Contents |
|---|---|
| `outputs/tickets_clean.csv` | Cleaned, enriched tickets with `dq_flags` |
| `outputs/classified_tickets.csv` | Classification cache (category, confidence, model, status: `ok` or `sim_reuse`) |
| `outputs/weekly_metrics.csv` | Weekly volume, category, SLA, cost, CSAT |
| `outputs/agent_weekly_metrics.csv` | Tier 1 weekly closures per agent |
| `outputs/agent_tier2_metrics.csv` | Tier 2 metrics, unranked |
| `outputs/repeat_contacts.csv` | Matched repeat pairs with evidence |
| `outputs/colleague_claim_analysis.json` | Transferred vs non-transferred comparison |
| `outputs/opportunities.csv` | Ranked opportunities |
| `outputs/labelling_sheet_50.csv` | 50-ticket sheet to label |
| `outputs/human_labels.csv` | Human ground-truth labels |
| `outputs/evaluation_metrics.json` | Evaluation on the human-labelled set |
| `outputs/digests_cache.json` | Cached weekly digests |
| `outputs/guardrail_stats.json` | Digest guardrail pass rate |
| `outputs/run_stats.json` | Last classification run |
| `outputs/embeddings.npy`, `embeddings_hashes.json` | Cached document embeddings (used by "Ask the Tickets") |

Prompts and decisions: `prompts/` (prompt files and `CHANGELOG.md`), `DECISIONS.md`, `docs/eda_findings.md`.

---

## 9. Known limitations

- **Small validation set.** 49 evaluated tickets, one labeller. The 69.4% figure has a wide interval (roughly 54–82%).
- **Similarity reuse.** 59% of cached labels are copied from a similar ticket, not classified by Gemma. The BM25 threshold (25) was not tuned, and reused rows scored lower in the small sample.
- **No retrieval few-shot in production.** The batch Gemma path sends no few-shot examples. Hybrid retrieval is used for "Ask the Tickets" and repeat-contact evidence, not for classification prompts.
- **Unclassified tickets.** 309 of the 11,875 unique tickets (2.6%) have no cached classification. In `tickets_clean.csv` they silently take the intake-bot category. One of the 50 labelled tickets is among them, which is why 49 were evaluated.
- **Confidence is a heuristic**, not a calibrated probability, and barely separates right from wrong.
- **Opportunity targets are assumptions** (for example 30% and 40% reductions), not benchmarks measured in the data.
- **Some digests are templates.** In `digests_cache.json`, 8 of 21 were written by the offline fallback, not Gemma. The 100% guardrail pass rate is inflated by this.
- **Repeat matching** is rule-based and was not checked against a hand-labelled sample of pairs.
- **Taxonomy discovery** used a random sample, not a stratified one.
- **Historical notes in `prompts/CHANGELOG.md`** (accuracy "~84%", "~4% JSON failures", "15–20x slower") were written before the human validation and have not been measured. Ignore them.
- **Dataset volume.** The data averages about 150 unique tickets a week, not the roughly 650 quoted in the brief.
- **`email-thread.txt`** must be copied into `data/` from the original pack.
- **Runtime** depends on hardware. Use `--limit` for smoke tests.

---

## 10. Tests

```bash
python -m pytest tests -q
```

14 tests cover SLA breach, contact, transfer and replacement costs, CSAT cleaning, timezone correction, weekly aggregation, Tier 2 exclusion and repeat-contact matching.
