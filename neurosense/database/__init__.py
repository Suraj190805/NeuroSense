"""NeuroSense Database — MongoDB Integration.

Async MongoDB backend using Motor for persistent storage of
users, predictions, cognitive sessions, and chat history.

Public API:
    connect_db()       — Initialize MongoDB connection
    close_db()         — Close MongoDB connection
    get_database()     — Get the Motor database instance
    crud               — CRUD operations module
    models             — Pydantic document models
"""

from neurosense.database.connection import close_db, connect_db, get_database
from neurosense.database.crud import (
    create_user,
    delete_chat_session,
    get_all_predictions,
    get_chat_history,
    get_prediction_by_id,
    get_predictions_by_user,
    get_sessions_by_patient,
    get_test_results,
    get_user_by_email,
    get_user_by_id,
    list_chat_sessions,
    list_users,
    save_chat_message,
    save_cognitive_session,
    save_prediction,
    save_test_result,
)
from neurosense.database.models import (
    ChatConversationDocument,
    CognitiveSessionDocument,
    PredictionDocument,
    TestResultDocument,
    UserDocument,
)

__all__ = [
    "connect_db",
    "close_db",
    "get_database",
    "create_user",
    "get_user_by_email",
    "get_user_by_id",
    "list_users",
    "save_prediction",
    "get_predictions_by_user",
    "get_prediction_by_id",
    "get_all_predictions",
    "save_cognitive_session",
    "get_sessions_by_patient",
    "save_chat_message",
    "get_chat_history",
    "list_chat_sessions",
    "delete_chat_session",
    "save_test_result",
    "get_test_results",
    "UserDocument",
    "PredictionDocument",
    "CognitiveSessionDocument",
    "ChatConversationDocument",
    "TestResultDocument",
]
