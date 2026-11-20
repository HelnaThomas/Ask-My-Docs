# Ask-My-Docs
Build a domain-specific "Ask My Docs" system with hybrid retrieval (BM25 + vector search), cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline. This is the most common pattern in enterprise AI right now.

ask-my-docs/
├── config/
│   └── prompts.yaml              # Versioned prompt architecture (Phase 2)
├── data/
│   └── raw_docs/                 # Your domain PDFs, Markdown, or text files
├── eval/
│   ├── golden_dataset.jsonl      # 50–200 verified QA pairs + unanswerable cases (Phase 3)
│   └── run_ragas_eval.py         # Offline & CI Ragas faithfulness gate (Phase 3)
├── src/
│   ├── __init__.py
│   ├── ingest.py                 # Chunking (500–800 tokens, 100 overlap) + Chroma indexing (Phase 1)
│   ├── retriever.py              # Hybrid BM25 + Chroma + Cross-Encoder reranker (Phase 2)
│   ├── generator.py              # Citation enforcement + explicit abstention (Phase 2)
│   ├── pipeline.py               # LangGraph orchestration + Langfuse tracing (Phase 1–4)
│   └── metrics.py                # P50/P95 latency, cost ($), citation coverage, failure rate (Ch. 4)
├── .github/
│   └── workflows/
│       └── ci-eval.yml           # Automated PR quality gate (Phase 3)
├── docker-compose.yml            # Self-hosted Langfuse observability stack (Ch. 4)
├── requirements.txt
└── app.py                        # CLI / API entry point

