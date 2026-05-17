SHL Assessment Recommendation — Approach Document

1. Design choices & retrieval setup

- Stateless message-based orchestration: Each `/chat` call accepts a `messages` array, the server reconstructs conversational state from these messages and performs retrieval each request. This avoids server-side session storage and improves reproducibility.
- Hybrid retrieval: BM25 (fast, deterministic) is the default for CI/tests (`USE_SEMANTIC_SEARCH=0`). Optional FAISS + sentence-transformer semantic search is supported when `USE_SEMANTIC_SEARCH=1` and the FAISS index is available.
- Shortlist derivation: The retriever returns catalog entries (name, link, categories, metadata). The API always returns catalog-bound URLs (or None if unavailable) to minimize hallucination.

2. Prompt design

- The system constructs plain-language explanation texts for each recommendation using `explanation_engine.py`. The prompt emphasizes catalog grounding, matched skills, and a short reasoning summary. No LLM blackbox instructions are embedded in server code.

3. Evaluation approach

- Replay harness: `tests/test_replay_harness.py` replays 10 official conversation traces (required to be placed in `traces/` or supplied as `traces.zip`) and validates schema compliance and catalog provenance.
- OpenAPI validation: `tests/test_openapi.py` verifies the `/openapi.json` contains `ChatRequest`/`ChatResponse` schemas and examples.
- CI: `.github/workflows/ci.yml` runs tests with semantic search disabled to avoid large model downloads.

4. What didn't work / limitations

- Models & FAISS: Semantic search requires large model downloads and index artifacts; CI and default tests disable it. If you need semantic search in CI, provide cached model artifacts and the `shl_index.faiss` file in the runner.
- Official traces: This repository doesn't contain the official 10 traces by default for licensing/privacy reasons. The test harness requires the evaluator to provide `traces.zip` or `traces/` with the 10 JSON traces.

5. AI tools used

- GitHub Copilot (Copilot Chat) assisted with iterative code edits, test-writing, and OpenAPI polishing. All final code changes are in this repository and reviewed.

Contact: platform@shl.com
