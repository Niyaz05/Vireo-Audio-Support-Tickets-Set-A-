# Prompt Engineering Changelog

## Versions

### `classify_v1.txt` (Current Production)
- **Role**: Support ticket classification using Ollama JSON schema enforcement.
- **Inputs**: `{CATEGORIES}`, `{PRODUCT}`, `{INTAKE_CATEGORY}`, `{FEW_SHOT}`, `{MESSAGE}`.
- **Format**: Structured JSON schema output (`category`: string enum, `confidence`: float).
- **Validation Metric**: Test set accuracy ~84%, macro-F1 ~0.82 (improves on intake-bot baseline of 68%).
- **Rationale**: Minimal tokens, enforces taxonomy directly in schema, includes intake bot as weak hint.

### `digest_v1.txt` (Current Production)
- **Role**: Executive weekly CX digest from deterministic JSON facts.
- **Headings**: Top complaint, What changed, Product signal, Repeat-contact signal, Cost signal, Worth investigating.
- **Guardrails**: Fact JSON injection, strict anti-hallucination constraint, regex number verification against source JSON.

### `discover_taxonomy_v1.txt` (Discovery)
- **Role**: Initial taxonomy discovery prompt across representative sample batches.
- **Output**: 12 consolidated categories stored in `src/config.py`.

---

## Discarded Approaches

1. **Free-form JSON generation without Ollama `format` parameter:**
   - *Why tried*: Simple prompt without schema.
   - *Why discarded*: Produced conversational filler ("Sure, here is your JSON...") and occasional markdown block wrapping that caused JSON decode exceptions on ~4% of queries. Replaced with native Ollama `format` JSON schema.

2. **Categorisation with Chain-of-Thought / Deep Thinking:**
   - *Why tried*: Tested thinking tokens for edge cases (Hinglish, mixed complaints).
   - *Why discarded*: Excessive latency (15-20x slower per ticket) with negligible F1 improvement on frontline tickets. Set `num_predict=100` and disabled thinking.

3. **Single monolithic vector search without BM25:**
   - *Why tried*: Pure dense embedding similarity using `nomic-embed-text`.
   - *Why discarded*: Missed exact model numbers (e.g. `VA-EB-PL2` vs `VA-EB-PL1`) and short numeric order lookups. Hybrid RRF (BM25 + Dense, k=60) significantly outperformed dense-only on ticket search recall.
