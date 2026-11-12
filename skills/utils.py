"""
utils.py — Shared helpers used by the reader, cleaner, analyzer and the app.
"""

import re

import pandas as pd

# Column-name tokens that mean "this is an identifier / code, not a measurement".
# Such columns must never be median-filled, outlier-trimmed or correlated.
_ID_LAST_TOKENS = {
    "id", "ids", "uuid", "guid", "no", "num", "number", "phone", "contact",
    "mobile", "cell", "cnic", "zip", "zipcode", "postal", "pincode", "sku",
    "ref", "code",
}
_ID_FIRST_TOKENS = {"id", "uuid", "guid"}


def _name_tokens(name) -> list[str]:
    """'StudentID' -> ['student', 'id'],  'contact_no' -> ['contact', 'no']."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def is_identifier_name(name) -> bool:
    """True if a column name looks like an ID / phone / code column."""
    tokens = _name_tokens(name)
    if not tokens:
        return False
    return tokens[-1] in _ID_LAST_TOKENS or tokens[0] in _ID_FIRST_TOKENS


def text_columns(df: pd.DataFrame) -> list:
    """
    Columns holding text, for any pandas version.
    (pandas >= 3 stores strings as the 'str' dtype, older versions use 'object'.)
    """
    cols = []
    for i, col in enumerate(df.columns):
        dtype = df.iloc[:, i].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            cols.append(col)
    return cols


def numeric_columns(df: pd.DataFrame, exclude_identifiers: bool = False) -> list:
    """Numeric (non-bool) columns, optionally without ID-like columns."""
    cols = []
    for i, col in enumerate(df.columns):
        dtype = df.iloc[:, i].dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            continue
        if exclude_identifiers and is_identifier_name(col):
            continue
        cols.append(col)
    return cols


def make_unique_columns(columns) -> list[str]:
    """Return string column names that are non-empty and unique."""
    seen: dict[str, int] = {}
    result = []
    for i, col in enumerate(columns):
        name = str(col).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        result.append(name)
    return result


def safe_filename(name: str) -> str:
    """Strip anything that is not safe in a file name."""
    base = re.sub(r"[^\w\-. ]+", "_", str(name)).strip(" .")
    return base or "data"
