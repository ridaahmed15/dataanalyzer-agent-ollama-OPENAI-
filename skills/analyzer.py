"""
analyzer.py — AI Analyst Skill
Handles: EDA, statistics, insights, business questions, dashboard suggestions
Uses OpenAI API for AI-powered analysis
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd
from openai import APIConnectionError, NotFoundError, OpenAI

from skills.utils import is_identifier_name, numeric_columns, text_columns


DEFAULT_OLLAMA_URL = "http://localhost:11434"


def ollama_openai_url(base_url: str | None = None) -> str:
    """Ollama serves an OpenAI-compatible API under /v1."""
    base = (base_url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def list_ollama_models(base_url: str | None = None, timeout: float = 3) -> list[str]:
    """Names of the models pulled in a running Ollama server ([] if it is not reachable)."""
    base = (base_url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """
    Create a client for OpenAI, or for any OpenAI-compatible server such as
    Ollama (pass base_url="http://localhost:11434/v1"; the key can be any text).
    """
    api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("OpenAI API key not found. Add it in the sidebar or in your .env file.")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=600)   # local models can be slow
    return OpenAI(api_key=api_key, timeout=60)


# ──────────────────────────────────────────────
# LOCAL STATISTICAL ANALYSIS (no API needed)
# ──────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> dict:
    """Return descriptive statistics for all columns."""
    stats = {}
    numeric_cols = numeric_columns(df)
    if numeric_cols:
        stats["numeric"] = df[numeric_cols].describe().round(2).to_dict()

    cat_cols = text_columns(df)
    if cat_cols:
        cat_stats = {}
        for col in cat_cols:
            vc = df[col].value_counts()
            cat_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "top_5": vc.head(5).to_dict(),
            }
        stats["categorical"] = cat_stats

    return stats


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix for numeric columns (ID-like columns are ignored)."""
    cols = numeric_columns(df, exclude_identifiers=True)
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr().round(3)


def top_correlations(df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Return pairs of columns with high correlation."""
    corr = correlation_analysis(df)
    if corr.empty:
        return []
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(val), 3),
                    "strength": "Strong" if abs(val) >= 0.75 else "Moderate",
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def quick_insights(df: pd.DataFrame) -> list[str]:
    """Generate rule-based insights without AI."""
    insights = []
    total = len(df)
    if total == 0:
        return ["⚠️ The dataset has no rows"]

    # Missing data
    missing_pct = (df.isnull().sum() / total * 100).round(1)
    for col, pct in missing_pct[missing_pct > 10].items():
        insights.append(f"⚠️ Column '{col}' has {pct}% missing values — consider review")

    # Numeric distribution (ID-like columns are not measurements)
    for col in numeric_columns(df, exclude_identifiers=True):
        skew = df[col].skew()
        if pd.notna(skew) and abs(skew) > 1.5:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            insights.append(f"📊 '{col}' is heavily skewed {direction} (skew={skew:.2f})")

    # Categorical cardinality
    for col in text_columns(df):
        values = df[col].dropna()
        if values.empty:
            continue
        n_unique = values.nunique()
        if n_unique == 1:
            insights.append(f"⚠️ '{col}' has only one unique value — may not be useful for analysis")
        elif n_unique == len(values) and len(values) > 1:
            insights.append(f"🔑 '{col}' appears to be a unique identifier (all values distinct)")
        elif n_unique <= 10:
            counts = values.value_counts()
            pct = round(counts.iloc[0] / len(values) * 100, 1)
            insights.append(f"📌 '{col}' top category: '{counts.index[0]}' ({pct}% of data)")

    # Correlation
    for pair in top_correlations(df, threshold=0.7)[:3]:
        insights.append(
            f"🔗 Strong correlation ({pair['correlation']}) between '{pair['col_a']}' and '{pair['col_b']}'"
        )

    return insights if insights else ["✅ No major data issues detected from quick scan"]


# ──────────────────────────────────────────────
# AI-POWERED ANALYSIS (OpenAI)
# ──────────────────────────────────────────────

def _group_summaries(df: pd.DataFrame, max_cats: int = 3, max_nums: int = 3, max_groups: int = 15) -> str:
    """
    count / mean / sum of the main numeric columns per category.
    Lets the model answer 'revenue by region' style questions from real numbers
    instead of guessing from three sample rows.
    """
    nums = numeric_columns(df, exclude_identifiers=True)[:max_nums]
    cats = [
        c for c in text_columns(df)
        if 2 <= df[c].nunique() <= max_groups and not is_identifier_name(c)
    ][:max_cats]
    if not nums or not cats:
        return ""
    blocks = ["Group summaries (count / mean / sum per category):"]
    for cat in cats:
        grouped = df.groupby(cat)[nums].agg(["count", "mean", "sum"]).round(2)
        blocks.append(f"\nBy {cat}:\n{grouped.to_string()}")
    return "\n".join(blocks)


def _build_data_context(df: pd.DataFrame) -> str:
    """Build a compact text summary of the DataFrame for the AI prompt."""
    numeric_cols = numeric_columns(df)
    cat_cols = text_columns(df)
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    lines = [
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(map(str, numeric_cols[:10]))}",
        f"Categorical columns ({len(cat_cols)}): {', '.join(map(str, cat_cols[:10]))}",
        f"Date columns ({len(date_cols)}): {', '.join(map(str, date_cols))}",
        f"Missing values: {int(df.isnull().sum().sum())} total",
    ]

    for col in date_cols:
        if df[col].notna().any():
            lines.append(f"Date range of '{col}': {df[col].min()} → {df[col].max()}")

    lines += ["", "Sample statistics (numeric):"]
    if numeric_cols:
        lines.append(df[numeric_cols[:6]].describe().round(2).to_string())

    if cat_cols:
        lines.append("\nTop categories:")
        for col in cat_cols[:4]:
            lines.append(f"  {col}: {df[col].value_counts().head(3).to_dict()}")

    pairs = top_correlations(df, threshold=0.5)[:5]
    if pairs:
        lines.append("\nTop correlations:")
        for p in pairs:
            lines.append(f"  {p['col_a']} ~ {p['col_b']}: {p['correlation']}")

    groups = _group_summaries(df)
    if groups:
        lines.append("\n" + groups)

    lines.append(f"\nFirst 3 rows:\n{df.iloc[:3, :12].to_string(index=False)}")

    return "\n".join(lines)


def _chat(prompt: str, model: str, temperature: float, max_tokens: int,
          api_key: str | None, base_url: str | None = None) -> str:
    client = get_openai_client(api_key, base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except APIConnectionError as e:
        if base_url:
            raise RuntimeError(
                "Could not reach Ollama. Start it with `ollama serve` and check the URL in the sidebar."
            ) from e
        raise
    except NotFoundError as e:
        if base_url:
            raise RuntimeError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}") from e
        raise
    return response.choices[0].message.content or ""


def ai_insights(df: pd.DataFrame, model: str = "gpt-4o-mini", api_key: str | None = None,
                base_url: str | None = None) -> str:
    """Use OpenAI to generate business insights from the data."""
    context = _build_data_context(df)

    prompt = f"""You are a senior data analyst. Analyze this dataset and provide:

1. KEY INSIGHTS (5-7 bullet points) — most important findings
2. TRENDS & PATTERNS — what the data reveals over time or across categories
3. ANOMALIES — anything unusual or worth investigating
4. BUSINESS RECOMMENDATIONS — 3-5 actionable suggestions based on the data

Dataset summary:
{context}

Be specific, use numbers from the data, and focus on business value.
Only state facts that are supported by the summary above."""

    return _chat(prompt, model, temperature=0.3, max_tokens=1200, api_key=api_key, base_url=base_url)


def ai_answer_question(df: pd.DataFrame, question: str, model: str = "gpt-4o-mini",
                       api_key: str | None = None, base_url: str | None = None) -> str:
    """Answer a specific business question about the data using AI."""
    context = _build_data_context(df)

    numeric_cols = numeric_columns(df)
    numeric_summary = (
        df[numeric_cols].agg(["sum", "mean", "min", "max"]).round(2).to_string()
        if numeric_cols else "No numeric columns"
    )

    prompt = f"""You are a data analyst. A user has uploaded a dataset and asked a question.
Answer the question accurately using the data provided. Show calculations where relevant.
If the summary below is not enough to answer precisely, say what is missing instead of guessing.

Dataset summary:
{context}

Numeric aggregates:
{numeric_summary}

User question: {question}

Provide a clear, direct answer with supporting evidence from the data."""

    return _chat(prompt, model, temperature=0.2, max_tokens=800, api_key=api_key, base_url=base_url)


def ai_dashboard_suggestions(df: pd.DataFrame, model: str = "gpt-4o-mini",
                             api_key: str | None = None, base_url: str | None = None) -> str:
    """Ask AI to suggest dashboard charts and KPIs."""
    context = _build_data_context(df)

    prompt = f"""You are a BI developer and data visualization expert.
Based on the dataset below, suggest a complete dashboard design.

For each suggestion include:
- Chart type (bar, line, pie, scatter, heatmap, etc.)
- X-axis and Y-axis (or dimensions)
- Why this chart is useful
- What business question it answers

Also suggest 4-6 KPI cards (key metrics to show at the top of the dashboard).

Dataset summary:
{context}

Format your response clearly with sections for: KPI Cards, Charts, and Filters."""

    return _chat(prompt, model, temperature=0.4, max_tokens=1000, api_key=api_key, base_url=base_url)
