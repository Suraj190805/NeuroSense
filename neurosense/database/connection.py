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

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# ─── Configuration ───
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017",
)
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "neurosense")

# ─── Global State ───
_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    """Initialize the MongoDB connection and create indexes.

    Connects the Motor async client and sets up collection
    indexes for efficient querying. Should be called once
    during application startup.

    Raises:
        ConnectionError: If MongoDB is unreachable.
    """
    global _client, _database

    try:
        # Use certifi CA bundle for SSL (fixes macOS certificate issues)
        import certifi

        _client = AsyncIOMotorClient(
            MONGODB_URI,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
            socketTimeoutMS=4000,
        )
        _database = _client[MONGODB_DB_NAME]

        # Verify connection
        await _client.admin.command("ping")
        logger.info(
            "MongoDB connected — uri=%s db=%s",
            MONGODB_URI,
            MONGODB_DB_NAME,
        )

        # Create indexes
        await _create_indexes()

    except Exception as e:
        logger.error("MongoDB connection failed: %s", e)
        _client = None
        _database = None
        raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e


async def close_db() -> None:
    """Close the MongoDB connection gracefully.

    Should be called during application shutdown.
    """
    global _client, _database

    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed")

    _client = None
    _database = None


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
    global _client, _database
    if _database is not None:
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
