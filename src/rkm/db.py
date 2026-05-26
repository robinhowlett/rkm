import os

import psycopg
from psycopg.rows import dict_row


def get_connection_string() -> str:
    if url := os.environ.get("RKM_DATABASE_URL"):
        return url
    host = os.environ.get("RKM_DB_HOST", "localhost")
    port = os.environ.get("RKM_DB_PORT", "5432")
    name = os.environ.get("RKM_DB_NAME", "handycapper")
    user = os.environ.get("RKM_DB_USER", "handycapper")
    password = os.environ.get("RKM_DB_PASSWORD", "handycapper")
    return f"host={host} port={port} dbname={name} user={user} password={password}"


def connect() -> psycopg.Connection:
    return psycopg.connect(get_connection_string(), row_factory=dict_row)


def connect_raw() -> psycopg.Connection:
    """Connection without dict_row — compatible with pandas read_sql."""
    return psycopg.connect(get_connection_string())
