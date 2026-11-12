"""
cleaner.py — Data Cleaner Skill
Handles: column names, whitespace / placeholder values, duplicates, type fixing,
missing values, outliers
"""

import os
import warnings

import numpy as np
import pandas as pd

from skills.utils import (
    is_identifier_name,
    make_unique_columns,
    numeric_columns,
    safe_filename,
    text_columns,
)

# Text values that really mean "missing"
NULL_TOKENS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing"}

# Only columns whose *name* says "date/time" are tried as dates
_DATE_NAME_TOKENS = {"date", "datetime", "timestamp", "time", "dob"}

OUTLIER_IQR_FACTOR = 3
MIN_ROWS_FOR_OUTLIERS = 10        # IQR on tiny samples is meaningless
MAX_OUTLIER_SHARE = 0.10          # >10% "outliers" = heavy tail, not errors
MIN_UNIQUE_FOR_OUTLIERS = 10      # skip flags / coded / discrete columns
CATEGORICAL_MODE_FILL_MAX_SHARE = 0.20   # more missing than this → 'Unknown', not the mode


def _looks_like_date_column(name: str) -> bool:
    import re
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    tokens = {t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t}
    return bool(tokens & _DATE_NAME_TOKENS)


def _to_numeric_series(values: pd.Series) -> pd.Series:
    """Parse '1,200', '$45', '12%' style strings; unparseable → NaN."""
    t = values.astype(str).str.strip()
    t = t.str.replace(r"^[\$€£]|%$", "", regex=True).str.strip()
    thousands = t.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
    t = t.where(~thousands, t.str.replace(",", "", regex=False))
    return pd.to_numeric(t, errors="coerce")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Full cleaning pipeline.
    Returns (cleaned_df, list_of_actions_taken)
    """
    steps: list[str] = []
    df = df.copy()

    # 1. Clean column names (also copes with non-string / duplicate / blank names)
    original_cols = df.columns.tolist()
    new_cols = make_unique_columns(
        pd.Index([str(c) for c in original_cols]).str.strip().str.replace(r"\s+", "_", regex=True)
    )
    df.columns = new_cols
    if new_cols != original_cols:
        steps.append("✅ Column names cleaned (whitespace stripped, spaces → underscores)")

    # 2. Strip whitespace + turn placeholder text ('N/A', '', 'null' …) into real missing values
    stripped_cells = 0
    placeholder_cells = 0
    touched_cols = 0
    for col in text_columns(df):
        s = df[col]
        has_ws = s.map(lambda v: isinstance(v, str) and v != v.strip())
        is_placeholder = s.map(lambda v: isinstance(v, str) and v.strip().lower() in NULL_TOKENS)
        if not (has_ws.any() or is_placeholder.any()):
            continue
        touched_cols += 1
        stripped_cells += int((has_ws & ~is_placeholder).sum())
        placeholder_cells += int(is_placeholder.sum())

        def _clean(v):
            if isinstance(v, str):
                v = v.strip()
                return np.nan if v.lower() in NULL_TOKENS else v
            return v          # numbers etc. are left untouched

        df[col] = s.map(_clean)
    if stripped_cells:
        steps.append(f"✅ Stripped extra whitespace from {stripped_cells} text cells")
    if placeholder_cells:
        steps.append(f"✅ Converted {placeholder_cells} placeholder values ('N/A', 'null', '', …) to missing")

    # 3. Drop completely empty rows / columns
    empty_cols = [c for c in df.columns if df[c].isna().all()] if len(df) else []
    if empty_cols:
        df = df.drop(columns=empty_cols)
        steps.append(f"✅ Dropped {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}")
    n_before = len(df)
    df = df.dropna(how="all")
    if len(df) < n_before:
        steps.append(f"✅ Dropped {n_before - len(df)} completely empty rows")

    # 4. Remove fully duplicate rows (after whitespace cleanup so 'A ' == 'A')
    dups = int(df.duplicated().sum())
    if dups > 0:
        df = df.drop_duplicates()
        steps.append(f"✅ Removed {dups} duplicate rows")
    df = df.reset_index(drop=True)

    # 5. Fix numeric columns stored as text (IDs / phone numbers / zero-padded codes stay text)
    for col in text_columns(df):
        if is_identifier_name(col):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        as_text = non_null.astype(str)
        if as_text.str.match(r"^0\d+").any():      # '007', '0300…' → codes, not numbers
            continue
        converted = _to_numeric_series(non_null)
        ratio = converted.notna().sum() / len(non_null)
        if ratio >= 0.9:
            lost = int(converted.isna().sum())
            df[col] = converted.reindex(df.index)
            msg = f"✅ Converted '{col}' to numeric"
            if lost:
                msg += f" ({lost} non-numeric values set to missing)"
            steps.append(msg)

    # 6. Fix date columns (by column name, and only if most values really parse)
    for col in text_columns(df):
        if not _looks_like_date_column(col):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        as_text = non_null.astype(str)
        if as_text.str.contains(r"\d").mean() < 0.8:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(as_text, errors="coerce", format="mixed")
        if parsed.notna().sum() / len(non_null) >= 0.8:
            df[col] = parsed.reindex(df.index)
            steps.append(f"✅ Converted '{col}' to datetime")

    # 7. Infinite values → missing
    num_cols = numeric_columns(df)
    if num_cols:
        inf_mask = np.isinf(df[num_cols].to_numpy(dtype=float, na_value=np.nan))
        if inf_mask.any():
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            steps.append(f"✅ Replaced {int(inf_mask.sum())} infinite values with missing")

    # 8. Handle missing values
    missing = df.isnull().sum()
    for col in df.columns:
        n_missing = int(missing[col])
        if n_missing == 0:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            steps.append(f"ℹ️ Left {n_missing} missing dates in '{col}' unfilled (not guessing dates)")
        elif pd.api.types.is_bool_dtype(series):
            fill_val = series.mode().iloc[0]
            df[col] = series.fillna(fill_val)
            steps.append(f"✅ Filled {n_missing} missing values in '{col}' with mode ({fill_val})")
        elif pd.api.types.is_numeric_dtype(series):
            if is_identifier_name(col):
                steps.append(f"ℹ️ Left {n_missing} missing values in ID-like column '{col}' unfilled")
                continue
            fill_val = series.median()
            df[col] = series.fillna(fill_val)
            steps.append(f"✅ Filled {n_missing} missing values in '{col}' with median ({fill_val:.2f})")
        else:
            mode = series.mode()
            # A mode fill is fine for a few gaps, but with many gaps it invents a
            # dominant category — use an explicit 'Unknown' instead.
            too_many_gaps = n_missing / len(series) > CATEGORICAL_MODE_FILL_MAX_SHARE
            fill_val = "Unknown" if (is_identifier_name(col) or mode.empty or too_many_gaps) else mode.iloc[0]
            df[col] = series.fillna(fill_val)
            steps.append(f"✅ Filled {n_missing} missing values in '{col}' with '{fill_val}'")

    # 9. Remove extreme outliers (IQR × 3) — measurement columns only
    if len(df) >= MIN_ROWS_FOR_OUTLIERS:
        outlier_mask = pd.Series(False, index=df.index)
        details = []
        for col in numeric_columns(df, exclude_identifiers=True):
            s = df[col]
            if s.nunique() <= MIN_UNIQUE_FOR_OUTLIERS:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            col_mask = (s < q1 - OUTLIER_IQR_FACTOR * iqr) | (s > q3 + OUTLIER_IQR_FACTOR * iqr)
            n_out = int(col_mask.sum())
            if n_out == 0:
                continue
            if n_out / len(df) > MAX_OUTLIER_SHARE:
                steps.append(f"ℹ️ '{col}' has {n_out} extreme values (>10% of rows) — kept, looks like a heavy-tailed column")
                continue
            outlier_mask |= col_mask
            details.append(f"{col}: {n_out}")
        if outlier_mask.any():
            df = df[~outlier_mask].reset_index(drop=True)
            steps.append(
                f"✅ Removed {int(outlier_mask.sum())} extreme outlier rows "
                f"(IQR × {OUTLIER_IQR_FACTOR} method; {', '.join(details)})"
            )

    if not any(s.startswith("✅") for s in steps):
        steps.insert(0, "✅ Data was already clean — no changes needed")

    return df.reset_index(drop=True), steps


def save_original(df: pd.DataFrame, filename: str, folder: str = "data/original") -> str:
    """Save the original (uncleaned) DataFrame as CSV."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, safe_filename(filename))
    df.to_csv(path, index=False)
    return path


def save_cleaned(df: pd.DataFrame, filename: str, folder: str = "data/cleaned") -> str:
    """Save the cleaned DataFrame as CSV."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, safe_filename(filename))
    df.to_csv(path, index=False)
    return path


def _count_duplicates(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:       # unhashable cells
        return 0


def get_cleaning_summary(original: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Compare original vs cleaned DataFrames."""
    return {
        "original_rows": len(original),
        "cleaned_rows": len(cleaned),
        "rows_removed": len(original) - len(cleaned),
        "original_columns": len(original.columns),
        "cleaned_columns": len(cleaned.columns),
        "original_missing": int(original.isnull().sum().sum()),
        "cleaned_missing": int(cleaned.isnull().sum().sum()),
        "original_duplicates": _count_duplicates(original),
        "cleaned_duplicates": _count_duplicates(cleaned),
    }
