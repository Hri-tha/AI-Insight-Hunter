"""
Database connection helper. Every backend module that needs Postgres calls
get_connection() rather than opening its own connection, so the pgvector
type is always registered correctly.
"""

import psycopg2
from pgvector.psycopg2 import register_vector

from backend.config import DATABASE_URL


def get_connection():
    """Opens a new Postgres connection with pgvector support enabled.

    register_vector() teaches psycopg2 how to convert a Python list of
    numbers into Postgres's `vector` column type, and back again when
    reading — without it, embeddings would fail to save or load.
    """
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn
