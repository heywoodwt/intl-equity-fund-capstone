"""Thin WRDS query layer.

The `wrds` client's own `raw_sql` is broken under pandas 2.x + SQLAlchemy 1.4:
it hands pandas a SQLAlchemy-1.4 ``Connection``, which pandas 2.x no longer
recognizes as a connectable, then falls back to calling ``.cursor()`` on it
(`'Connection' object has no attribute 'cursor'`). We instead query through the
engine's raw psycopg2 connection, which pandas reads fine. Use `query()` for all
data pulls; `describe_table`/`list_tables` on the wrds object are unaffected.
"""

import os
import warnings

import pandas as pd
import wrds


def connect(username=None):
    """Open a WRDS connection (username from arg or $WRDS_USERNAME; password
    from ~/.pgpass)."""
    username = username or os.environ.get("WRDS_USERNAME")
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def query(db, sql, params=None):
    """Run SQL on a wrds connection and return a DataFrame, bypassing the broken
    `raw_sql` path via the raw DBAPI (psycopg2) connection."""
    raw = db.engine.raw_connection()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_sql_query(sql, raw, params=params)
    finally:
        raw.close()
