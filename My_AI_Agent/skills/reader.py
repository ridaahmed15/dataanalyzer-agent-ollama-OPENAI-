"""
reader.py — Data Reader Skill
Supports: CSV, Excel, SQLite/SQL, JSON
"""

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from skills.utils import is_identifier_name, make_unique_columns

_NESTED_TYPES = (list, dict, set, tuple)
_CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


# ──────────────────────────────────────────────
# Individual readers
# ──────────────────────────────────────────────

def read_csv(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.
    Tries several encodings (UTF-8 with/without BOM, Windows-1252, Latin-1) and
    falls back to delimiter sniffing for ';', tab or '|' separated files.
    """
    last_error = None
    for encoding in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except pd.errors.ParserError:
            df = pd.read_csv(filepath, encoding=encoding, sep=None, engine="python")

        # A single column usually means the wrong delimiter was assumed
        sep = ","
        if df.shape[1] == 1:
            try:
                sniffed = pd.read_csv(filepath, encoding=encoding, sep=None, engine="python")
                if sniffed.shape[1] > 1:
                    df, sep = sniffed, None
            except Exception:
                pass

        return _restore_leading_zeros(df, filepath, encoding, sep)

    raise ValueError(f"Could not decode the CSV file: {last_error}")


def _restore_leading_zeros(df: pd.DataFrame, filepath: str, encoding: str, sep) -> pd.DataFrame:
    """
    pandas turns '0300123456' into the number 300123456 and the zero is lost.
    For ID / phone / code columns, re-read them as text when that happens.
    """
    id_cols = [
        c for c in df.columns
        if is_identifier_name(c) and pd.api.types.is_numeric_dtype(df[c].dtype)
    ]
    if not id_cols:
        return df
    try:
        kwargs = {"engine": "python"} if sep is None else {}
        raw = pd.read_csv(filepath, encoding=encoding, sep=sep, usecols=id_cols,
                          dtype={c: str for c in id_cols}, **kwargs)
    except Exception:
        return df
    for col in id_cols:
        if raw[col].dropna().str.match(r"^0\d+").any():
            df[col] = raw[col]
    return df


def read_excel(filepath: str, sheet_name=0) -> pd.DataFrame:
    """Read an Excel file (.xlsx / .xls) and return a DataFrame."""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def read_json(filepath: str) -> pd.DataFrame:
    """
    Read a JSON file and return a DataFrame.
    Handles: list of records, {"data": [records]}, column-oriented dicts,
    JSON Lines, and nested records (flattened with json_normalize).
    """
    with open(filepath, "r", encoding="utf-8-sig") as f:
        text = f.read()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # maybe JSON Lines: one object per line
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}") from e

    if isinstance(data, dict):
        record_lists = [
            v for v in data.values()
            if isinstance(v, list) and v and isinstance(v[0], dict)
        ]
        if record_lists:
            data = record_lists[0]
        elif data and all(isinstance(v, list) for v in data.values()):
            lengths = {len(v) for v in data.values()}
            if len(lengths) == 1:                      # {"col": [..], "col2": [..]}
                return pd.DataFrame(data)
            data = [data]
        else:
            data = [data]

    if isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            return pd.json_normalize(data)
        return pd.DataFrame({"value": data})

    return pd.DataFrame({"value": [data]})


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def read_sql(db_path: str, query: str = None) -> pd.DataFrame:
    """
    Connect to a SQLite database (read-only) and execute a query.
    If no query is given, reads the first table.
    """
    with closing(_connect_readonly(db_path)) as conn:
        if query is None:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if tables.empty:
                raise ValueError("No tables found in the database.")
            first_table = str(tables.iloc[0]["name"]).replace('"', '""')
            return pd.read_sql(f'SELECT * FROM "{first_table}"', conn)

        if not query.lstrip().lower().startswith(("select", "with")):
            raise ValueError("Only SELECT queries are allowed.")
        return pd.read_sql(query, conn)


def get_sql_tables(db_path: str) -> list:
    """Return list of table names in a SQLite database."""
    with closing(_connect_readonly(db_path)) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    return tables["name"].tolist()


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Make any freshly-read DataFrame safe for the rest of the pipeline."""
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError("The file was read successfully but contains no data.")

    df = df.copy()
    df.columns = make_unique_columns(df.columns)

    # ID / phone columns that became float only because of missing values → whole numbers
    for col in df.columns:
        if is_identifier_name(col) and pd.api.types.is_float_dtype(df[col].dtype):
            values = df[col].dropna()
            if not values.empty and (values % 1 == 0).all():
                df[col] = df[col].astype("Int64")

    # Lists / dicts inside cells (from JSON) are unhashable and break
    # duplicated(), nunique(), value_counts() … → store them as JSON text.
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            is_nested = series.map(lambda v: isinstance(v, _NESTED_TYPES))
            if is_nested.any():
                df[col] = series.map(
                    lambda v: json.dumps(v, default=str) if isinstance(v, _NESTED_TYPES) else v
                )
    return df


def read_file(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Auto-detect file type and read accordingly.
    Supported: .csv, .xlsx, .xls, .json, .db, .sqlite
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".csv":
        df = read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = read_excel(filepath, sheet_name=kwargs.get("sheet_name", 0))
    elif ext == ".json":
        df = read_json(filepath)
    elif ext in (".db", ".sqlite"):
        df = read_sql(filepath, query=kwargs.get("query", None))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return _finalize(df)


def get_basic_info(df: pd.DataFrame) -> dict:
    """Return basic metadata about a DataFrame."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample": df.head(5).to_dict(orient="records"),
    }
