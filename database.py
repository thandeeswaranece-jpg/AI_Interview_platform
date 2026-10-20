import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "users.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def create_user(username, email, password_hash):
    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO users
            (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, email, password_hash)
        )

        connection.commit()

        user_id = cursor.lastrowid

        return {
            "id": user_id,
            "username": username,
            "email": email
        }

    except sqlite3.IntegrityError:
        return None

    finally:
        connection.close()


def get_user_by_email(email):
    connection = get_connection()

    user = connection.execute(
        """
        SELECT id, username, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    connection.close()

    return dict(user) if user else None


def get_user_by_id(user_id):
    connection = get_connection()

    user = connection.execute(
        """
        SELECT id, username, email
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    connection.close()

    return dict(user) if user else None