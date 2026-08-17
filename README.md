# Finance Insight Lite

Finance Insight Lite is a local financial document analysis app. It lets you upload financial reports, spreadsheets, CSV files, and supported image files, then ask questions over the uploaded content using a Retrieval-Augmented Generation (RAG) pipeline powered by Groq and LangChain.

The main experience is a Streamlit chat interface with source references, confidence metadata, persistent chat history, and Plotly visualizations for finance-oriented questions.

## What It Does

- Answers questions about uploaded financial documents.
- Retrieves relevant report pages, table rows, and extracted chunks before generating an answer.
- Supports PDF, Excel, CSV, PNG, JPG, and JPEG uploads.
- Builds a local vector database with Chroma/FAISS-style retrieval.
- Uses hybrid retrieval, query expansion, corrective RAG grading, and optional answer verification.
- Generates charts for questions that ask to visualize financial metrics.
- Persists chat history in a local SQLite database.
- Exposes both a Streamlit UI and a FastAPI service.

## Current Status

This is an active prototype for financial RAG workflows. It is designed for local experimentation, demos, and evaluation against financial reports such as Saudi Aramco filings. It is not a production financial advice system.

## Requirements

- Python 3.11 or newer
- A Groq API key
- macOS, Linux, or Windows
- At least 4 GB RAM, with 8 GB or more recommended for larger reports

The project includes both `requirements.txt` and `uv.lock`. Use whichever workflow fits your environment.

## Setup

Clone or open the repository, then create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or, if you use `uv`:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

## Run the Streamlit App

```bash
streamlit run src/ui.py
```

Then open:

```text
http://localhost:8501
```

In the sidebar, upload one or more supported files and click **Process All Documents**. Once processing finishes, ask questions in the chat box.

## Run the FastAPI Service

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Useful endpoints:

- `GET /health` - check service status
- `POST /upload` - upload and process one document
- `POST /query` - ask a question against the loaded document
- `GET /document/info` - inspect the currently loaded document
- `DELETE /document` - clear the loaded document
- `GET /docs` - OpenAPI documentation

## Example Questions

- What is the net income?
- What is the free cash flow?
- What is the gearing ratio?
- How much was the dividend declared?
- What are the key financial highlights?
- Draw a bar chart of quarterly revenue.
- Visualize profit margins.
- Reconcile the ledger against the bank statement.

## Supported Inputs

| Type | Extensions | Notes |
| --- | --- | --- |
| PDF reports | `.pdf` | Text is extracted with PyMuPDF. Embedded images are inspected when OCR support is available. |
| Excel workbooks | `.xlsx`, `.xls` | Sheets are converted into retrievable table documents. |
| CSV files | `.csv` | Rows and small tables are preserved for structured-table retrieval. |
| Images | `.png`, `.jpg`, `.jpeg` | Intended for screenshots of financial tables or report images. |

Image and embedded-image extraction uses OCR-related libraries from the processing module when available. If image processing fails in your environment, use PDF, Excel, or CSV inputs first, then install the missing OCR dependencies shown in the error output.

## Project Layout

```text
.
|-- main.py                                # FastAPI application
|-- src/
|   |-- ui.py                              # Streamlit application
|   |-- app.py                             # Combined API/UI launcher draft
|   |-- chat_db.py                         # SQLite chat-history helpers
|   `-- finance_insight_lite/
|       `-- modules/
|           |-- processor.py               # PDF, table, image, cache processing
|           |-- verctor_store.py           # Vector database construction
|           |-- rag_agent.py               # Main RAG orchestration
|           |-- hybrid_retriever.py        # Semantic plus keyword retrieval
|           |-- query_expansion.py         # Query expansion
|           |-- structured_tables.py       # Table-aware document helpers
|           |-- workflow_coordinator.py    # Task routing and workflow hints
|           |-- groq_client.py             # Groq integration helpers
|           `-- eval.py                    # Evaluation runner
|-- data/
|   |-- rew/                               # Example raw financial files
|   |-- uploaded/                          # Files uploaded through the UI
|   |-- processed/                         # Processed markdown/text outputs
|   `-- vector_db/                         # Local vector index files
|-- database/                              # Local app vector database
|-- tests/                                 # Unit and evaluation tests
|-- skills/                                # Role/workflow prompt assets
|-- requirements.txt
|-- pyproject.toml
`-- uv.lock
```

## Core Pipeline

1. **Document processing**
   `processor.py` loads PDFs, spreadsheets, CSVs, and images into LangChain `Document` objects. It preserves useful metadata such as source file, page, row, and extraction type.

2. **Structured-table handling**
   `structured_tables.py` keeps small tables intact where possible, so ledger, budget, revenue operations, and reconciliation questions can use full table context.

3. **Vector indexing**
   `verctor_store.py` builds the local retrieval index used by the app. The filename currently keeps the existing project spelling.

4. **Retrieval and grading**
   `rag_agent.py` combines adaptive retrieval depth, hybrid retrieval, query expansion, and corrective RAG grading to select context for the answer.

5. **Answer generation**
   The agent calls Groq-hosted chat models through LangChain, generates a finance-oriented answer, and returns source pages, confidence, verification metadata, and optional chart data.

6. **Visualization**
   Chart requests are detected from the query. The app extracts tabular/numeric data and renders Plotly bar, line, pie, or area charts.

## Configuration

The main required environment variable is:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional LangSmith tracing:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

The Streamlit sidebar also exposes runtime controls:

- Self-RAG verification toggle
- Relevance threshold
- Number of retrieved documents
- Default chart type
- Data table display toggle
- Cache clearing
- Chat history clearing

## Data and Persistence

- Uploaded files are stored under `data/uploaded/`.
- Local vector indexes are written under `database/` or `data/vector_db/`, depending on the entry point.
- Chat history is stored in `src/chat_history.db`.
- Cached processing artifacts may be stored under `data/cache/`.

These local artifacts can grow over time. Clear them when you want a fresh run or need to reclaim disk space.

## Tests and Evaluation

Run the test suite:

```bash
python -m pytest
```

Export chat history for evaluation:

```bash
python scripts/export_chat_history.py
```

Run the financial RAG judge:

```bash
python -m finance_insight_lite.modules.eval \
  --input tests/fixtures/chat_history_export.jsonl \
  --output tests/output/judge_results.json \
  --with-judge
```

The judge path may require model credentials and network access.

## Troubleshooting

### The app says no document is loaded

Upload files in the Streamlit sidebar and click **Process All Documents** before asking questions.

### Groq requests fail

Check that `.env` exists in the project root and contains a valid `GROQ_API_KEY`. Also confirm your Groq account has available quota.

### Image upload or embedded-image extraction fails

Image OCR relies on optional runtime imports in `processor.py`. If your workflow does not need image OCR, use PDF, Excel, or CSV files. If you do need it, install the missing OCR/OpenCV dependencies reported by the traceback.

### Responses get slow

Large documents, many retrieved chunks, answer verification, and long chat sessions can increase latency. Try lowering the number of retrieved documents, disabling Self-RAG, clearing chat history, or processing fewer files at once.

### Charts do not appear

Charts require extractable numeric data. Try asking for a specific metric and period, for example: `Draw a bar chart of net income by quarter`.

## Notes for Contributors

- Prefer keeping financial logic in `src/finance_insight_lite/modules/`.
- Preserve metadata when adding new document loaders; retrieval and source attribution depend on it.
- Keep UI state small before passing chat history into the LLM. Raw chunks and Plotly JSON can quickly bloat prompts.
- Avoid installing `uvicorn[standard]` on macOS for this app; the project intentionally uses plain `uvicorn` because `uvloop` can conflict with Streamlit.

## License

No license file is currently included in this repository. Add one before distributing or reusing this project outside private development.
