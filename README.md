# 🤖 Smart Data Analyst AI Agent

A full-featured AI-powered data analysis agent built with Python + Streamlit.

## ✅ Features

| Feature | Description |
|---|---|
| 📄 Read CSV / Excel / JSON | Auto-detect and load any tabular data |
| 🗄️ Read SQL / SQLite | Connect and query .db / .sqlite databases |
| 💾 Save Original Data | Automatically saves untouched copy |
| 🧹 Auto Clean Data | Nulls, duplicates, type fixes, outliers |
| 📁 Export Cleaned CSV | Clean data saved separately |
| 📊 EDA & Charts | Distributions, correlations, time series |
| 💡 Auto Insights | Rule-based insight engine |
| 🤖 AI Insights | Ollama (local) or OpenAI business analysis |
| ❓ Business Q&A | Ask questions in plain English |
| 📊 Dashboard Ideas | AI suggests charts and KPIs |
| 📄 Full Report | Save .txt report to /reports folder |

## 📁 Project Structure

```
My_AI_Agent/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env.example            # copy to .env and add your API key (never commit .env)
├── skills/
│   ├── reader.py           # CSV, Excel, JSON, SQL reader
│   ├── cleaner.py          # Data cleaning pipeline
│   ├── utils.py            # Shared helpers (column-type detection, safe filenames)
│   ├── analyzer.py         # Stats + OpenAI analysis
│   └── report.py           # Report generator
├── data/
│   ├── original/           # Auto-saved original files
│   ├── cleaned/            # Auto-saved cleaned files
│   └── sample_sales.csv    # Test dataset
└── reports/                # Generated .txt reports
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
cd My_AI_Agent
pip install -r requirements.txt
```

### 2. Choose an AI provider

**Option A — Ollama (free, local, offline) — default**
```bash
ollama serve                 # skip if it already runs in the background
ollama pull llama3.1         # or qwen2.5, mistral, gemma2 …
```
Then pick **Ollama (local, free)** in the sidebar — installed models appear automatically.
If your Ollama is on another machine, set its URL in the sidebar or `OLLAMA_BASE_URL` in `.env`.
Tip: for bigger datasets give the model more context, e.g. `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`.

**Option B — OpenAI**
Copy `.env.example` to `.env` and edit it:
```
OPENAI_API_KEY=sk-your-key-here
```
Or paste it directly in the app sidebar.

### 3. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

## 🧪 Test with sample data
A sample sales CSV is included at `data/sample_sales.csv`.
Upload it from the sidebar to try all features.

## 💡 How to Use

1. **Upload** a CSV, Excel, JSON, or SQLite file from the sidebar
2. Click **🚀 Load & Analyze** — data is auto-read, cleaned, and saved
3. Browse the **6 tabs**: Overview → Cleaning → EDA → Insights → Ask AI → Report
4. Click **💡 AI Insights** for AI analysis (needs Ollama running, or an OpenAI key)
5. Use **❓ Ask AI** tab to ask business questions in plain English
6. Click **📄 Generate Full Report** to create a downloadable report
"# dataanalyzer-agent-ollama-OPENAI-" 
