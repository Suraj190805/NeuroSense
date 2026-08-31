"""NeuroSense Database — MongoDB Connection Management.

Manages the async Motor client lifecycle for use with FastAPI's
lifespan events. The connection is configured via environment
variables for flexibility across dev/staging/production.

Environment Variables:
    MONGODB_URI      — Connection string (default: mongodb://localhost:27017)
    MONGODB_DB_NAME  — Database name (default: neurosense)

Usage:
    # In FastAPI lifespan
    async def lifespan(app):
        await connect_db()
        yield
        await close_db()

    # In route handlers
    db = get_database()
    result = await db.patients.find_one({"patient_id": "P001"})
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# ─── Load Environment Variables ───
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ─── Global State ───
_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None
_connected_uri: str | None = None


def _get_target_config() -> tuple[str, str]:
    """Retrieve current MongoDB URI and DB name from environment."""
    try:
        from dotenv import load_dotenv

        _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if _env_path.exists():
            load_dotenv(dotenv_path=_env_path, override=True)
    except ImportError:
        pass

    uri = os.getenv(
        "MONGODB_URI",
        "mongodb://localhost:27017",
    )
    db_name = os.getenv("MONGODB_DB_NAME", "neurosense")
    return uri, db_name


async def connect_db(force: bool = False) -> None:
    """Initialize the MongoDB connection and create indexes.

    Connects the Motor async client and sets up collection
    indexes for efficient querying. Should be called once
    during application startup.

    Raises:
        ConnectionError: If MongoDB is unreachable.
    """
    global _client, _database, _connected_uri

    uri, db_name = _get_target_config()

    # Skip reconnect if already connected to the same URI unless forced
    if not force and _database is not None and _connected_uri == uri:
        return

    # Close previous client if URI changed
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
        _database = None
        _connected_uri = None

    try:
        client_kwargs = {
            "serverSelectionTimeoutMS": 6000,
            "connectTimeoutMS": 6000,
            "socketTimeoutMS": 6000,
        }

        # Apply TLS/certifi only when connecting over TLS / mongodb+srv
        if "mongodb+srv" in uri or "ssl=true" in uri.lower() or "tls=true" in uri.lower():
            try:
                import certifi

                client_kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass

        new_client = AsyncIOMotorClient(uri, **client_kwargs)
        new_db = new_client[db_name]

        # Verify connection
        await new_client.admin.command("ping")

        _client = new_client
        _database = new_db
        _connected_uri = uri

        masked_host = uri.split("@")[-1] if "@" in uri else uri
        logger.info(
            "MongoDB connected successfully — host=%s db=%s",
            masked_host,
            db_name,
        )

        # Create indexes
        await _create_indexes()

    except Exception as e:
        logger.error("MongoDB connection failed: %s", e)
        _client = None
        _database = None
        _connected_uri = None
        raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e


async def close_db() -> None:
    """Close the MongoDB connection gracefully.

    Should be called during application shutdown.
    """
    global _client, _database, _connected_uri

    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed")

    _client = None
    _database = None
    _connected_uri = None


def get_database() -> AsyncIOMotorDatabase:
    """Get the active database instance.

    Returns:
        The Motor database object.

    Raises:
        RuntimeError: If the database is not connected.
    """
    if _database is None:
        raise RuntimeError(
            "Database not connected. Call connect_db() first."
        )
    return _database


def is_connected() -> bool:
    """Check if the database connection is active.

    Returns:
        True if connected, False otherwise.
    """
    return _database is not None


async def ensure_connected() -> bool:
    """Ensure MongoDB connection is established; auto-connects if disconnected.

    Returns:
        True if connected or reconnected successfully, False otherwise.
    """
    global _client, _database, _connected_uri
    uri, _ = _get_target_config()

    if _database is not None and _connected_uri == uri:
        return True
    try:
        await connect_db()
        return _database is not None
    except Exception as e:
        logger.error("MongoDB auto-reconnect failed: %s", e)
        return False


async def _create_indexes() -> None:
    """Create collection indexes for query performance.

    Indexes:
        users: email (unique), createdAt
        predictions: userId + createdAt
        cognitive_sessions: patient_id + created_at, session_id
        chat_conversations: session_id + timestamp
    """
    if _database is None:
        return

    # Users collection
    users = _database.users
    await users.create_index("email", unique=True)
    await users.create_index("createdAt")

    # Predictions collection
    predictions = _database.predictions
    await predictions.create_index(
        [("userId", 1), ("createdAt", -1)]
    )

    # Cognitive sessions collection
    sessions = _database.cognitive_sessions
    await sessions.create_index(
        [("patient_id", 1), ("created_at", -1)]
    )
    await sessions.create_index("session_id", unique=True)

    # Chat conversations collection
    chats = _database.chat_conversations
    await chats.create_index(
        [("session_id", 1), ("timestamp", -1)]
    )

    logger.info("MongoDB indexes created")
