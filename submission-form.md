# Submission Form: Vireo Audio Support Tickets (Set A)

---

## 1. What did you build, and what business outcome does it move?

I built a local Streamlit app that cleans 18 months of Vireo tickets (12,528 rows, 11,875 unique after flagging 653 re-imports). It produces:

- a weekly complaint digest
- a Tier 1 "tickets closed per week" leaderboard
- a repeat-contact analysis
- a ranked table of savings opportunities

Python computes every number. Gemma 4 through Ollama only classifies and writes text.

**The data-chosen top opportunity is DOA-driven replacements: cut the rate from 1.1% to 0.8% of tickets, worth about Rs 1,37,139 a quarter.**

This is a scenario resting on an assumed 30% reduction, not a forecast. Evidence grade is B, because the DOA count is observed but the link to supplier QC is estimated.

Transfers (9.1% of tickets, about Rs 30,700 a quarter) and SLA breaches (8.9%, about Rs 28,800) are better measured but smaller.

---

## 2. What does one run cost, and what would a month cost at Vireo's volume (roughly 650 tickets a week)?

I made **no paid calls**. Everything runs locally, so API cost is Rs 0. Local electricity and hardware were not measured.

**Monthly volume:** 650 × 52 ÷ 12 ≈ **2,817 tickets**

The last recorded run took 2,322 s for 267 Gemma calls (batches of up to 8), which is about 8.7 s per call.

| Scenario | Arithmetic | Time per month |
|---|---|---|
| Worst case, every ticket needs Gemma | 2,817 ÷ 8 ≈ 352 calls × 8.7 s | about 51 minutes |
| With today's 59% reuse rate | about 1,150 tickets to Gemma, so about 144 calls × 8.7 s | about 21 minutes |
| Weekly digests | 4 Gemma calls | negligible |

The dataset itself averages about 150 unique tickets a week, not 650, so this is conservative.

---

## 3. How do you know it works?

I labelled 50 tickets by hand (stratified sample), without looking at Gemma's answers. 49 could be scored, because one has no cached classification.

| Metric | Result |
|---|---|
| Sample size | 49 scored (50 labelled) |
| Accuracy | **69.4%** (34 of 49) |
| Error rate | 30.6% |
| Macro-F1 | 59.5% |
| Intake-bot baseline accuracy | 32.7% |
| Plausible range for accuracy | roughly 54–82% |

With n = 49 this is a rough guide, not a precise figure.

**Cases it gets wrong most often:**

- `Other`: 0 of 2 correct
- `Physical Damage & Build` mistaken for `Warranty & Repair`: 0 of 2 correct
- `Product Enquiry`: recall 33%
- `Returns & Refunds` (the largest class): recall 58%

**Other checks:**

- Confidence scores don't help. 46 of 49 tickets sit at 0.85 or above, with 69.6% accuracy there.
- Rows Gemma classified itself scored 78.9% (15 of 19). Copied rows scored 63.3% (19 of 30).
- 14 passing pytest tests cover the deterministic logic (SLA, costs, CSAT, timezone, tier split, repeat rules).

---

## 4. Did you change, narrow, or push back on the client's ask?

- **Leaderboard narrowed.** Priya asked for agents ranked by tickets closed. I ranked Tier 1 only, labelled it "not a performance score", and gave Tier 2 Warranty an unranked view. Neha asked not to rank her team on counts, and policy §6 measures Tier 2 in days.
- **Colleague claim tested, not assumed.** Transferred tickets repeat slightly less (3.8%) than non-transferred (4.6%), so the data doesn't support "I already told your colleague".
- **Goal chosen from data.** I ranked eight candidates and didn't assume repeat contacts were the prize.
- **Per-channel costs used (Rs 210–520).** Arjun said Rs 180, Priya said Rs 290, and policy §4 gives per-channel figures.
- **Validation narrowed to 50 labels.** I planned about 220 and cut to 50 to fit the five-hour cap.

---

## 5. What is wrong with what you are handing us?

- The validation set is small (49 scored, one labeller), so the accuracy is uncertain by roughly ±14 points.
- 59% of classifications are copied from similar tickets by a BM25 threshold I never tuned, and those scored lower in my sample.
- The production batch path uses no few-shot examples, so hybrid retrieval doesn't help classification. It is only used for "Ask the Tickets" and repeat evidence.
- 309 tickets (2.6%) have no classification and silently use the intake-bot tag.
- Opportunity targets are assumed reductions (30–40%). Any "industry benchmark" wording has no source.
- 8 of 21 cached digests are fallback templates, so the 100% guardrail pass rate is inflated.
- The repeat-contact matcher was not checked against hand-labelled pairs.
- Taxonomy discovery used a random sample rather than a stratified one.
- `prompts/CHANGELOG.md` and `DECISIONS.md` still contain claims from before validation (84% accuracy, "validated on labelled set") that I never measured.

---

## 6. What did you deliberately leave out, and why that rather than something else?

I left out:

- a larger validation set
- tuning the reuse threshold
- few-shot retrieval in the batch path
- checking the repeat matcher by hand
- any forecasting

I put correctness of the numbers, the tier split and honest validation ahead of those. Forecasting was left out on purpose, because without a deployed change there is no evidence for a saving.

---

## 7. Anything you built or found that nobody asked for?

- **653 re-imported Freshdesk tickets** that would have double-counted refunds and volume.
- **A timezone bug.** Legacy `resolved_at` was off by exactly 5.5 hours on 618 of 618 overlapping tickets, which made 60% of legacy tickets show a resolution before creation.
- **2 refund-plus-replacement policy violations** (Rs 8,955).
- **A number-checking guardrail on digests**, and an "Ask the Tickets" search page.

---

## 8. What did you use AI for?

**Tools and models**

- **Claude (chat):** wrote the build prompt and reviewed the finished project.
- **Antigravity IDE (Claude Opus 4.6, via Gemini Student membership):** wrote most of the code.
- **Gemma 4 via Ollama:** runs inside the app for classification and digests, at no cost.

**Where it helped:** scaffolding, the pandas logic, the policy-rule constants, and the test suite.

**Where it wasted my time:** the first version silently faked its classifications from the intake-bot tag, generated its own "human labels" from keyword rules, hardcoded baseline accuracy numbers, and wrote an unmeasured "84% accuracy" into the changelog. I found this in a review.

**What I threw away:** all of the above. I replaced it with real Gemma output and my own 50 hand labels.

**Screen recording (3 minutes):** https://drive.google.com/file/d/1hZek4Gq6e1waquXavOKcCC3KwUsErYXV/view?usp=sharing

---

## 9. Your Public Google Drive Link

https://drive.google.com/file/d/1hZek4Gq6e1waquXavOKcCC3KwUsErYXV/view?usp=sharing

---

## 10. Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **The accuracy is 69.4% on only 49 hand-labelled tickets (plausible range about 54–82%), and 59% of labels are copied from similar tickets rather than classified by Gemma.** Tune the reuse threshold and relabel a larger sample before anyone relies on the complaint categories.
2. **The Rs 1,37,139 a quarter is a scenario built on an assumed 30% cut in DOA replacements.** Get a real DOA baseline from Finance and QC before quoting it.
3. **To run it:** copy the pack into `data/`, start `ollama serve`, run `python -m src.pipeline`, then `streamlit run app.py`. The dashboard works from cached outputs without Ollama, except "Ask the Tickets".

---

## 11. Honest hours spent

**6**

---

## 12. GitHub Repo Link

https://github.com/Niyaz05/Vireo-Audio-Support-Tickets-Set-A-