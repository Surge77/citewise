# Citewise

Answers questions from your PDFs, with the page to prove it.

Citewise is a retrieval-augmented generation (RAG) app. Drop PDFs into `data/`, ask a question in the browser, and get
a short answer where every claim cites the document and page it came from. If your documents don't contain the
answer, it says so instead of guessing.

## How it works

```
PDF ─► pages ─► sentences ─► deduplicated, page-bounded chunks ─► MiniLM embeddings ─► Chroma (on disk)

question ─┬─► meaning search (Chroma, top 20) ─┐
          └─► keyword search (BM25, top 20) ───┴─► reciprocal rank fusion ─► cross-encoder rerank ─► top 4
                                                                                         │
            best meaning similarity < 0.15? ── yes ─► "I couldn't find this in your documents." (no LLM call)
                                              no ──► Gemini 2.5 Flash, numbered sources ─► answer with [n] citations
```

- **Chunking** splits on sentence boundaries, never crosses a page (so citations are exact), and drops repeated
  sentences anywhere in the document.
- **Hybrid search** combines meaning search (good at paraphrases) with BM25 keyword search (good at exact terms like
  `HAVING` or `DAX`). Results are merged by rank, so the two scoring scales never need to agree.
- **Reranking** reads the question and each candidate passage together with a cross-encoder, which judges relevance
  more accurately than comparing two vectors.
- **Refusal** is decided before calling the LLM: off-topic questions cost nothing and can't produce a made-up answer.

## Retrieval evaluation

`eval/run_eval.py` scores retrieval only (no LLM calls) on 20 answerable and 6 off-topic questions written for a
10-page data-analytics notes PDF:

| Metric | Meaning only | Hybrid | **Hybrid + rerank** |
|---|---|---|---|
| hit@1 (right page ranked first) | 0.90 | 0.70 | **1.00** |
| hit@4 | 0.95 | 1.00 | **1.00** |
| MRR | 0.93 | 0.82 | **1.00** |
| Off-topic questions refused | 1.00 | 1.00 | **1.00** |

Hybrid search alone *lowered* first-place accuracy (BM25 matches filler words on a tiny corpus) but put every
right page in the top 4; the reranker then ordered them correctly. The question set is small and was written
alongside the code, so treat these numbers as a regression baseline, not a benchmark. The sample PDF is not
included in this repo; to evaluate your own documents, put them in `data/` and write `eval/questions.jsonl` lines
like `{"question": "...", "page": 3}` (use `"page": null` for questions the documents can't answer).

## Quickstart

Requires Python 3.11+ and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
set GEMINI_API_KEY=your-key       # Windows  (macOS/Linux: export GEMINI_API_KEY=your-key) — or put it in .env
```

Put one or more PDFs in `data/`, then:

```bash
python -m citewise                # serves http://127.0.0.1:8000
python -m eval.run_eval           # retrieval scores for your documents
pytest --cov                      # tests; no models or network needed
```

The first start downloads two small models (~180 MB total) from Hugging Face. If the download stalls on Windows,
set `HF_HUB_DISABLE_XET=1`. PDFs are indexed on startup; a file that is already indexed (same content hash) is
skipped.

## API

`POST /api/ask` with `{"question": "..."}` (1–500 characters):

```json
{
  "success": true,
  "data": {
    "answer": "The SQL keyword that filters grouped results is HAVING [1].",
    "found": true,
    "citations": [{"number": 1, "doc_name": "notes.pdf", "page": 7, "snippet": "SQL is used to retrieve…"}]
  },
  "timestamp": "2026-10-27T10:00:00+00:00"
}
```

Errors use the same envelope with `"success": false` and a code: `VALIDATION_ERROR` (422, with per-field
messages), `LLM_UNAVAILABLE` (502) or `INTERNAL_ERROR` (500). Error responses never include internal details.

## Project layout

| Path | Responsibility |
|---|---|
| `citewise/ingest.py` | Read a PDF into numbered pages; content-hash document ids |
| `citewise/chunking.py` | Sentence splitting, deduplication, page-bounded chunks |
| `citewise/store.py` | Chroma vector store (on disk, or in memory for evaluation) |
| `citewise/keyword.py` | BM25 keyword index |
| `citewise/retriever.py` | Reciprocal rank fusion and the hybrid retriever |
| `citewise/reranker.py` | Cross-encoder reranker |
| `citewise/answer.py` | Prompt with numbered sources, citation parsing, refusal |
| `citewise/llm.py` | Gemini client with clean error mapping |
| `citewise/api.py` | FastAPI app and response envelope |
| `static/` | Chat page (all model output rendered as text, never HTML) |
| `eval/` | Retrieval evaluation set and scorer |

## Configuration

Defaults live in `citewise/config.py`: embedding model `all-MiniLM-L6-v2`, reranker `ms-marco-MiniLM-L-6-v2`,
LLM `gemini-2.5-flash`, 4 sources per answer, 20 candidates per search, similarity floor 0.15, chunks up to 600
characters. Re-tune the similarity floor with `eval/run_eval.py --min-similarity` when your documents change.
