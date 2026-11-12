"""
report.py — Report Generator Skill
Generates a full text report and saves it to the /reports folder
"""

import os
from datetime import datetime
import pandas as pd

from skills.utils import safe_filename


def generate_report(
    filename: str,
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    cleaning_steps: list[str],
    cleaning_summary: dict,
    local_insights: list[str],
    ai_insights: str = "",
    ai_dashboard: str = "",
    output_folder: str = "reports",
) -> tuple[str, str]:
    """
    Compile all analysis into a structured text report.
    Returns (path_of_saved_report, report_text).
    """
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = safe_filename(os.path.splitext(os.path.basename(filename))[0])
    report_path = os.path.join(output_folder, f"{base_name}_report_{timestamp}.txt")

    lines = []

    # ── Header ──────────────────────────────────
    lines += [
        "=" * 70,
        "           SMART DATA ANALYST AI AGENT — FULL REPORT",
        "=" * 70,
        f"  File       : {filename}",
        f"  Generated  : {datetime.now().strftime('%B %d, %Y  %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    # ── Section 1: Dataset Overview ─────────────
    lines += [
        "━" * 70,
        "  SECTION 1: DATASET OVERVIEW",
        "━" * 70,
        f"  Original shape  : {original_df.shape[0]} rows × {original_df.shape[1]} columns",
        f"  Cleaned shape   : {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns",
        f"  Rows removed    : {cleaning_summary.get('rows_removed', 0)}",
        f"  Missing (before): {cleaning_summary.get('original_missing', 0)}",
        f"  Missing (after) : {cleaning_summary.get('cleaned_missing', 0)}",
        "",
        "  Columns:",
    ]
    for col in cleaned_df.columns:
        dtype = str(cleaned_df[col].dtype)
        null_count = int(cleaned_df[col].isnull().sum())
        lines.append(f"    • {col:<30} [{dtype}]   nulls: {null_count}")
    lines.append("")

    # ── Section 2: Cleaning Steps ────────────────
    lines += [
        "━" * 70,
        "  SECTION 2: DATA CLEANING STEPS",
        "━" * 70,
    ]
    for step in cleaning_steps:
        lines.append(f"  {step}")
    lines.append("")

    # ── Section 3: Descriptive Statistics ────────
    lines += [
        "━" * 70,
        "  SECTION 3: DESCRIPTIVE STATISTICS (CLEANED DATA)",
        "━" * 70,
    ]
    numeric_df = cleaned_df.select_dtypes(include="number")
    if not numeric_df.empty:
        lines.append(numeric_df.describe().round(2).to_string())
    else:
        lines.append("  No numeric columns found.")
    lines.append("")

    # ── Section 4: Local Insights ─────────────────
    lines += [
        "━" * 70,
        "  SECTION 4: AUTOMATED INSIGHTS",
        "━" * 70,
    ]
    for insight in local_insights:
        lines.append(f"  {insight}")
    lines.append("")

    # ── Section 5: AI Insights ────────────────────
    if ai_insights:
        lines += [
            "━" * 70,
            "  SECTION 5: AI-POWERED ANALYSIS (OpenAI)",
            "━" * 70,
            ai_insights,
            "",
        ]

    # ── Section 6: Dashboard Suggestions ─────────
    if ai_dashboard:
        lines += [
            "━" * 70,
            "  SECTION 6: DASHBOARD SUGGESTIONS",
            "━" * 70,
            ai_dashboard,
            "",
        ]

    # ── Footer ────────────────────────────────────
    lines += [
        "=" * 70,
        "  END OF REPORT — Smart Data Analyst AI Agent",
        "=" * 70,
    ]

    report_text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path, report_text


def save_qa_log(
    questions: list[dict],
    base_filename: str,
    output_folder: str = "reports",
) -> str:
    """Save Q&A session log to a separate file."""
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = safe_filename(os.path.splitext(os.path.basename(base_filename))[0])
    log_path = os.path.join(output_folder, f"{base_name}_qa_log_{timestamp}.txt")

    lines = [
        "=" * 70,
        "  BUSINESS Q&A SESSION LOG",
        "=" * 70,
        "",
    ]
    for i, item in enumerate(questions, 1):
        lines += [
            f"Q{i}: {item.get('question', '')}",
            "-" * 40,
            item.get("answer", ""),
            "",
        ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return log_path
