"""NeuroSense Database — CRUD Operations.

Async CRUD functions for all MongoDB collections. Each function
uses the Motor async driver and handles serialization between
Pydantic models and MongoDB documents.

All functions are designed to be called from FastAPI async
route handlers without blocking the event loop.

Functions:
    Users:
        create_user()               — Register a new user
        get_user_by_email()         — Get user by email
        get_user_by_id()            — Get user by MongoDB _id
        list_users()                — List all users

    Predictions:
        save_prediction()           — Save a prediction result
        get_prediction_by_id()      — Get prediction by _id
        get_predictions_by_user()   — Get all predictions for a user
        get_all_predictions()       — Get all predictions

    Cognitive Sessions:
        save_cognitive_session()    — Save a completed session
        get_sessions_by_patient()   — Get all sessions for a patient
        get_session_by_id()         — Get session by ID

    Chat:
        save_chat_message()         — Append messages to a conversation
        get_chat_history()          — Get conversation by session ID
        list_chat_sessions()        — List all chat sessions
        delete_chat_session()       — Delete a chat session
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from neurosense.database.connection import get_database
from neurosense.database.models import (
    ChatConversationDocument,
    ChatMessageDoc,
    CognitiveSessionDocument,
    PredictionDocument,
    TestResultDocument,
    UserDocument,
)

logger = logging.getLogger(__name__)


# ─── Helper ───

def _serialize_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert MongoDB _id (ObjectId) to string for JSON serialization."""
    if doc is None:
        return None
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def _serialize_list(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert _id in a list of documents."""
    return [_serialize_doc(d) for d in docs]


# ═════════════════════════════════════════════════════════════════
#  Users
# ═════════════════════════════════════════════════════════════════


async def create_user(user: UserDocument) -> str:
    """Register a new user.

    Args:
        user: User document to insert.

    Returns:
        The string _id of the created user.

    Raises:
        ValueError: If a user with this email already exists.
    """
    db = get_database()

    # Check for duplicate email
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise ValueError(
            f"User with email '{user.email}' already exists"
        )

    doc = user.model_dump()
    result = await db.users.insert_one(doc)

    user_id = str(result.inserted_id)
    logger.info("Created user: %s (%s)", user.name, user.email)
    return user_id


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Get a user by email address.

    Args:
        email: The user's email.

    Returns:
        User document dict with _id as string, or None.
    """
    db = get_database()
    doc = await db.users.find_one({"email": email})
    return _serialize_doc(doc)


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    """Get a user by MongoDB _id.

    Args:
        user_id: The user's _id as string.

    Returns:
        User document dict with _id as string, or None.
    """
    db = get_database()
    try:
        doc = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    return _serialize_doc(doc)


async def list_users(
    skip: int = 0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List all users, paginated.

    Args:
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of user document dicts (password excluded).
    """
    db = get_database()
    cursor = (
        db.users.find({}, {"password": 0})
        .sort("createdAt", -1)
        .skip(skip)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return _serialize_list(docs)


# ═════════════════════════════════════════════════════════════════
#  Predictions (embedded inside user documents)
# ═════════════════════════════════════════════════════════════════


async def save_prediction(prediction: PredictionDocument) -> str:
    """Save a prediction result.

    For logged-in users: pushes into the user's embedded
    ``predictions`` array — keeping all data under one document.

    For anonymous users: inserts into the standalone
    ``predictions`` collection as a fallback.

    Args:
        prediction: Prediction document to insert.

    Returns:
        The prediction ID (either the generated predictionId or
        the standalone collection _id).
    """
    db = get_database()
    doc = prediction.model_dump()
    user_id = doc.get("userId", "anonymous")

    if user_id and str(user_id).strip() not in ("anonymous", "null", "undefined", "None", ""):
        # ── Embed inside user document ──
        import uuid

        pred_id = str(uuid.uuid4())[:12]
        doc["predictionId"] = pred_id

        try:
            if ObjectId.is_valid(str(user_id)):
                result = await db.users.update_one(
                    {"_id": ObjectId(str(user_id))},
                    {"$push": {"predictions": doc}},
                )

                if result.modified_count > 0:
                    logger.info(
                        "Saved prediction %s (embedded in user=%s, prediction=%s, risk=%s)",
                        pred_id, user_id,
                        prediction.prediction, prediction.riskLevel,
                    )
                    return pred_id
                else:
                    logger.warning(
                        "User %s not found in users collection, saving prediction to standalone collection",
                        user_id,
                    )
        except Exception as e:
            logger.warning(
                "Could not embed prediction in user %s: %s",
                user_id, e
            )

    # ── Standalone collection (anonymous / fallback) ──
    doc["userId"] = str(user_id) if user_id else "anonymous"
    result = await db.predictions.insert_one(doc)
    pred_id = str(result.inserted_id)
    logger.info(
        "Saved prediction: %s (standalone, user=%s, prediction=%s, risk=%s)",
        pred_id, user_id,
        prediction.prediction, prediction.riskLevel,
    )
    return pred_id


async def get_prediction_by_id(
    prediction_id: str,
    user_id: str | None = None,
) -> dict[str, Any] | None:
    """Get a single prediction by its ID.

    Searches the user's embedded predictions first, then falls
    back to the standalone predictions collection.

    Args:
        prediction_id: The prediction identifier.
        user_id: Optional user ID to search embedded predictions.

    Returns:
        Prediction document dict, or None if not found.
    """
    db = get_database()

    # Try embedded predictions first
    if user_id and str(user_id).strip() not in ("anonymous", "null", "undefined", "None", ""):
        try:
            if ObjectId.is_valid(str(user_id)):
                user = await db.users.find_one(
                    {"_id": ObjectId(str(user_id))},
                    {"predictions": 1},
                )
                if user and "predictions" in user:
                    for pred in user["predictions"]:
                        if pred.get("predictionId") == prediction_id or str(pred.get("_id")) == prediction_id:
                            pred["_id"] = prediction_id
                            return pred
        except Exception:
            pass

    # Search all users for embedded prediction
    try:
        user = await db.users.find_one(
            {"predictions.predictionId": prediction_id},
            {"predictions.$": 1},
        )
        if user and "predictions" in user and user["predictions"]:
            pred = user["predictions"][0]
            pred["_id"] = prediction_id
            return pred
    except Exception:
        pass

    # Fallback to standalone collection
    try:
        if ObjectId.is_valid(prediction_id):
            doc = await db.predictions.find_one(
                {"_id": ObjectId(prediction_id)}
            )
            if doc:
                return _serialize_doc(doc)
        # Search by string ID or predictionId
        doc = await db.predictions.find_one(
            {"$or": [{"_id": prediction_id}, {"predictionId": prediction_id}]}
        )
        if doc:
            return _serialize_doc(doc)
    except Exception:
        pass

    return None


async def get_predictions_by_user(
    user_id: str,
    skip: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get all predictions for a user, most recent first.

    Reads from the user's embedded predictions array. Falls back
    to the standalone collection if the user has no embedded data.

    Args:
        user_id: The user's _id as string.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of prediction document dicts.
    """
    db = get_database()

    # Read embedded predictions from user document
    if user_id and str(user_id).strip() not in ("anonymous", "null", "undefined", "None", ""):
        try:
            if ObjectId.is_valid(str(user_id)):
                user = await db.users.find_one(
                    {"_id": ObjectId(str(user_id))},
                    {"predictions": 1},
                )
                if user and "predictions" in user and user["predictions"]:
                    predictions = user["predictions"]
                    # Add _id field for frontend compatibility
                    for pred in predictions:
                        if "predictionId" in pred:
                            pred["_id"] = pred["predictionId"]
                    # Sort by createdAt descending
                    predictions.sort(
                        key=lambda p: str(p.get("createdAt", "")),
                        reverse=True,
                    )
                    return predictions[skip : skip + limit]
        except Exception as e:
            logger.warning("Failed to read embedded predictions: %s", e)

    # Fallback to standalone collection
    try:
        cursor = (
            db.predictions.find({"userId": str(user_id)})
            .sort("createdAt", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return _serialize_list(docs)
    except Exception as e:
        logger.warning("Failed to read standalone predictions: %s", e)
        return []


async def get_all_predictions(
    skip: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get all predictions across all users.

    Aggregates embedded predictions from all user documents
    plus any standalone predictions.

    Args:
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of prediction document dicts.
    """
    db = get_database()
    all_preds: list[dict[str, Any]] = []

    # Gather embedded predictions from all users
    try:
        cursor = db.users.find(
            {"predictions": {"$exists": True, "$ne": []}},
            {"predictions": 1, "name": 1, "email": 1},
        )
        async for user in cursor:
            uid = str(user["_id"])
            for pred in user.get("predictions", []):
                pred["userId"] = uid
                if "predictionId" in pred:
                    pred["_id"] = pred["predictionId"]
                all_preds.append(pred)
    except Exception as e:
        logger.warning("Failed to aggregate embedded predictions: %s", e)

    # Also include standalone predictions
    try:
        standalone_cursor = (
            db.predictions.find({})
            .sort("createdAt", -1)
            .limit(limit)
        )
        standalone = await standalone_cursor.to_list(length=limit)
        all_preds.extend(_serialize_list(standalone))
    except Exception:
        pass

    # Sort all by createdAt descending and paginate
    all_preds.sort(
        key=lambda p: str(p.get("createdAt", "")),
        reverse=True,
    )
    return all_preds[skip : skip + limit]


# ═════════════════════════════════════════════════════════════════
#  Cognitive Sessions
# ═════════════════════════════════════════════════════════════════


async def save_cognitive_session(
    session: CognitiveSessionDocument,
) -> str:
    """Save a completed cognitive session.

    Args:
        session: Session document to insert.

    Returns:
        The session_id of the saved record.
    """
    db = get_database()
    doc = session.model_dump()
    await db.cognitive_sessions.insert_one(doc)

    logger.info(
        "Saved cognitive session: %s (patient=%s)",
        session.session_id,
        session.patient_id,
    )
    return session.session_id


async def get_session_by_id(
    session_id: str,
) -> dict[str, Any] | None:
    """Get a cognitive session by ID.

    Args:
        session_id: The session identifier.

    Returns:
        Session document dict, or None if not found.
    """
    db = get_database()
    doc = await db.cognitive_sessions.find_one(
        {"session_id": session_id},
    )
    return _serialize_doc(doc)


async def get_sessions_by_patient(
    patient_id: str,
    skip: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get all cognitive sessions for a patient.

    Args:
        patient_id: The patient identifier.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of session document dicts, most recent first.
    """
    db = get_database()
    cursor = (
        db.cognitive_sessions.find({"patient_id": patient_id})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return _serialize_list(docs)


# ═════════════════════════════════════════════════════════════════
#  Chat Conversations
# ═════════════════════════════════════════════════════════════════


async def save_chat_message(
    session_id: str,
    user_message: str,
    assistant_reply: str,
    patient_id: str = "anonymous",
) -> str:
    """Save a chat exchange (user + assistant) to a conversation.

    If the conversation session doesn't exist, creates it.
    If it exists, appends the new messages.

    Args:
        session_id: Conversation session identifier.
        user_message: The user's message text.
        assistant_reply: The assistant's reply text.
        patient_id: Optional patient identifier.

    Returns:
        The session_id.
    """
    db = get_database()
    now = datetime.now(timezone.utc)

    user_msg = ChatMessageDoc(
        role="user",
        content=user_message,
        timestamp=now,
    )
    assistant_msg = ChatMessageDoc(
        role="assistant",
        content=assistant_reply,
        timestamp=now,
    )

    # Upsert: create if not exists, append messages if exists
    result = await db.chat_conversations.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        user_msg.model_dump(),
                        assistant_msg.model_dump(),
                    ]
                }
            },
            "$inc": {"message_count": 2},
            "$set": {
                "updated_at": now,
                "patient_id": patient_id,
                "last_user_message": user_message,
                "last_assistant_reply": assistant_reply,
            },
            "$setOnInsert": {
                "session_id": session_id,
                "created_at": now,
            },
        },
        upsert=True,
    )

    logger.debug(
        "Saved chat exchange: session=%s (upserted=%s)",
        session_id,
        result.upserted_id is not None,
    )
    return session_id


async def get_chat_history(
    session_id: str,
) -> dict[str, Any] | None:
    """Get a full chat conversation by session ID.

    Args:
        session_id: The conversation session identifier.

    Returns:
        Conversation document dict, or None if not found.
    """
    db = get_database()
    doc = await db.chat_conversations.find_one(
        {"session_id": session_id},
    )
    return _serialize_doc(doc)


async def list_chat_sessions(
    patient_id: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List chat sessions, optionally filtered by patient.

    Args:
        patient_id: Optional filter by patient.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of conversation summary dicts (without full messages).
    """
    db = get_database()

    query: dict[str, Any] = {}
    if patient_id:
        query["patient_id"] = patient_id

    cursor = (
        db.chat_conversations.find(
            query,
            {
                "_id": 0,
                "session_id": 1,
                "patient_id": 1,
                "message_count": 1,
                "last_user_message": 1,
                "last_assistant_reply": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
        .sort("updated_at", -1)
        .skip(skip)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def delete_chat_session(session_id: str) -> bool:
    """Delete a chat conversation by session ID.

    Args:
        session_id: The conversation session identifier.

    Returns:
        True if a document was deleted, False if not found.
    """
    db = get_database()
    result = await db.chat_conversations.delete_one(
        {"session_id": session_id}
    )
    if result.deleted_count > 0:
        logger.info("Deleted chat session: %s", session_id)
        return True
    return False


# ═════════════════════════════════════════════════════════════════
#  Test Results (Memory & Motor Tests)
# ═════════════════════════════════════════════════════════════════


async def save_test_result(result: TestResultDocument) -> str:
    """Save a test result to the database.

    Embeds into user's document if patientId is a valid user,
    and also persists into the standalone ``test_results`` collection.

    Args:
        result: Test result document to insert.

    Returns:
        The string _id of the saved result.
    """
    db = get_database()
    doc = result.model_dump()
    patient_id = doc.get("patientId", "anonymous")

    if patient_id and str(patient_id).strip() not in ("anonymous", "null", "undefined", "None", ""):
        try:
            if ObjectId.is_valid(str(patient_id)):
                await db.users.update_one(
                    {"_id": ObjectId(str(patient_id))},
                    {"$push": {"tests": doc}},
                )
        except Exception as e:
            logger.warning("Could not embed test result in user %s: %s", patient_id, e)

    res = await db.test_results.insert_one(doc)
    result_id = str(res.inserted_id)
    logger.info(
        "Saved test result: %s (patient=%s, test=%s, score=%s/%s)",
        result_id,
        patient_id,
        result.testName,
        result.score,
        result.total,
    )
    return result_id


async def get_test_results(
    patient_id: str | None = None,
    category: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get test results, optionally filtered by patient or category.

    Args:
        patient_id: Optional filter by patient.
        category: Optional filter by test category ('memory' or 'motor').
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of test result document dicts, most recent first.
    """
    db = get_database()

    query: dict[str, Any] = {}
    if patient_id:
        query["patientId"] = patient_id
    if category:
        query["testCategory"] = category

    cursor = (
        db.test_results.find(query)
        .sort("createdAt", -1)
        .skip(skip)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return _serialize_list(docs)
