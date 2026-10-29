# Citewise

[![CI](https://github.com/Surge77/citewise/actions/workflows/ci.yml/badge.svg)](https://github.com/Surge77/citewise/actions/workflows/ci.yml)

Answers questions from your PDFs, with the page to prove it.

Citewise is a retrieval-augmented generation (RAG) app. Upload PDFs in the browser (or drop them into `data/`), ask a question, and get
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

**Whole-library mode.** Search only finds passages that match a question, so it can't answer "summarize this PDF",
"list every topic", "how many pages" or "compare these two documents". When the whole library fits in one prompt
(60,000 characters by default, `full_context_chars`), Citewise skips search and sends every page as a numbered
source, so any kind of question works and every claim still cites its page. Larger libraries use the search path
above. In this mode off-topic questions are refused by the model (one LLM call) rather than by the similarity gate.

**Follow-ups.** The chat page sends the last 6 turns with each question, so "explain that more simply" or "give an
example" refer back to the conversation. In search mode the previous question is added to the search query.

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

`POST /api/ask/stream` takes the same body and streams newline-delimited JSON (`application/x-ndjson`) so text
appears as it is generated: `{"type": "delta", "text": "..."}` lines, then exactly one final line, either
`{"type": "done", "answer": ..., "found": ..., "citations": [...]}` or `{"type": "error", "code": ..., "message": ...}`.
Citations arrive only in the `done` line, because a `[n]` marker can be split across deltas. The response status
is always 200 once streaming starts; failures after that point are reported in the `error` line. The chat page
uses this endpoint.

`POST /api/documents?name=report.pdf` with the raw PDF as the body (`Content-Type: application/pdf`, up to
20 MB) indexes it and saves it to `data/`. Returns 201 with `{doc_id, name, pages, chunks_added, already_indexed,
empty_pages}`, or 200 if the same file (by content hash) is already indexed. `empty_pages` lists pages with no
extractable text, which usually means scanned images. `GET /api/documents` lists indexed documents.

Uploads are checked for the `%PDF-` signature, read with a hard size cap (the body is never fully buffered past
the limit), and saved under a sanitised file name with no directory components.

Errors use the same envelope with `"success": false` and a code: `VALIDATION_ERROR` (422, with per-field
messages), `INVALID_PDF` (400), `FILE_TOO_LARGE` (413), `LLM_UNAVAILABLE` (502) or `INTERNAL_ERROR` (500).
Error responses never include internal details.

## Project layout

| Path | Responsibility |
|---|---|
| `citewise/ingest.py` | Read a PDF into numbered pages; content-hash document ids |
| `citewise/chunking.py` | Sentence splitting, deduplication, page-bounded chunks |
| `citewise/store.py` | Chroma vector store (on disk, or in memory for evaluation) |
| `citewise/keyword.py` | BM25 keyword index |
| `citewise/retriever.py` | Reciprocal rank fusion and the hybrid retriever |
| `citewise/reranker.py` | Cross-encoder reranker |
| `citewise/library.py` | Upload validation, safe file names, saving and indexing uploads |
| `citewise/answer.py` | Prompt with numbered sources, citation parsing, refusal |
| `citewise/llm.py` | Gemini client with clean error mapping |
| `citewise/api.py` | FastAPI app and response envelope |
| `static/` | Chat page (all model output rendered as text, never HTML) |
| `eval/` | Retrieval evaluation set and scorer |

## Configuration

Defaults live in `citewise/config.py`: embedding model `all-MiniLM-L6-v2`, reranker `ms-marco-MiniLM-L-6-v2`,
LLM `gemini-2.5-flash`, 4 sources per answer, 20 candidates per search, similarity floor 0.15, chunks up to 600
characters. Re-tune the similarity floor with `eval/run_eval.py --min-similarity` when your documents change.
