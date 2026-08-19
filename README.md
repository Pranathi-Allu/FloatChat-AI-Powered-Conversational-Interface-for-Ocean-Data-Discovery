# FloatChat-AI-Powered-Conversational-Interface-for-Ocean-Data-Discovery

FloatChat is a natural-language interface for exploring ARGO ocean float data. Ask a question in plain English, and the app translates it into a SQL query, runs it against a PostgreSQL database, and automatically renders the results as maps, charts, or time-series visualizations.

---

## ✨ Features

- **Natural language → SQL**: Ask questions like *"Show average temperature in the Arabian Sea in 2020"* and get a valid PostgreSQL query generated automatically.
- **Local LLM inference**: Uses an Ollama-served LLaMA model — no external API calls, fully self-hosted.
- **Metadata-aware querying**: A FAISS vector index built from ARGO metadata PDFs grounds the query generation in the actual dataset schema and context.
- **Automatic visualization**: Query results are auto-mapped to the most relevant chart type — histogram, bar chart, scatter plot, or time-series — based on the shape of the returned data.
- **Geospatial mapping**: Plots float positions on an interactive map using PyDeck, with optional convex hull (Minimum Convex Polygon) overlays to show spatial coverage.
- **Interactive controls**: Fine-tune latitude/longitude columns, buffer distances, point radius, and chart axes directly from the sidebar.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────┐      ┌──────────────────┐
│  Streamlit UI    │─────▶│  LLM (Ollama /    │
│  (Text Input)    │      │  LLaMA 3.1)       │
└─────────────────┘      └──────────────────┘
                                  │
                                  ▼
                         Generated SQL Query
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   PostgreSQL DB   │
                        │  (argo_data table)│
                        └──────────────────┘
                                  │
                                  ▼
                          Query Results (df)
                                  │
                  ┌───────────────┼────────────────┐
                  ▼               ▼                ▼
             PyDeck Map     Altair Charts     Data Table
```

FAISS + metadata PDF are used to ground schema/context understanding (RAG-style retrieval) that supports the SQL generation prompt.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App | Streamlit |
| LLM Inference | Ollama (LLaMA 3.1 8B, abliterated) |
| Vector Store | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Orchestration | LangChain |
| Database | PostgreSQL (via SQLAlchemy + psycopg2) |
| Geospatial | GeoPandas, Shapely, PyDeck |
| Visualization | Altair, PyDeck |
| PDF Parsing | PyPDF2 |

---

## 📦 Prerequisites

- Python 3.10+
- PostgreSQL running locally (or accessible remotely) with your ARGO data loaded into a table named `argo_data`
- [Ollama](https://ollama.com) installed and running locally, with the model pulled:
  ```bash
  ollama pull mannix/llama3.1-8b-abliterated
  ```
- A metadata PDF (`floaat_merged.pdf`) describing the dataset, placed in the project root (used to build the FAISS index on first run)

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pranathi-Allu/FloatChat-AI-Powered-Conversational-Interface-for-Ocean-Data-Discovery.git
cd FloatChat-AI-Powered-Conversational-Interface-for-Ocean-Data-Discovery
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` yet, generate one with:
> ```bash
> pip freeze > requirements.txt
> ```

### 4. Configure environment variables

Create a `.env` file in the project root (this file is git-ignored and should **never** be committed):

```env
POSTGRES_USER=your_postgres_username
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_database_name
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 5. Start Ollama

```bash
ollama serve
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Enter a natural-language question in the text box (e.g., *"What is the average salinity below 500m depth in 2021?"*).
2. Click **Generate & Run Query**.
3. Review the auto-generated SQL, the result table, and the auto-selected visualizations.
4. Adjust map/chart options in the sidebar to customize the view.

---
