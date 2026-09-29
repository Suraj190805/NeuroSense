"""NeuroSense API — AI Chatbot Endpoints.

FastAPI router for the HD-focused AI chatbot powered by Groq.

Endpoints:
    POST /chatbot/chat
        Send a message and receive an AI response about
        Huntington's Disease, clinical queries, and NeuroSense usage.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from neurosense.database.connection import ensure_connected, is_connected
from neurosense.database.crud import (
    delete_chat_session,
    get_chat_history,
    list_chat_sessions,
    save_chat_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# ─── Groq Configuration ───
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv(
    "GROQ_API_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = [
    GROQ_MODEL,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
]
MODELS_TO_TRY = list(dict.fromkeys([m for m in FALLBACK_MODELS if m]))

# ─── System prompt: domain expert on Huntington's Disease ───
SYSTEM_PROMPT = """You are NeuroSense AI Assistant — a high-precision, empathetic clinical support assistant specialising in Huntington's Disease (HD), clinical symptomatology, pharmacotherapy, and the NeuroSense platform.

### 1. CLINICAL SYMPTOMATOLOGY (MAYO CLINIC & NIH / NINDS STANDARDS)
You possess expert mastery of Huntington's Disease clinical presentation across all three classic domains:

A. **Movement Disorders (Motor Domain)**:
   - **Chorea**: Involuntary, irregular, rapid, dance-like jerking or writhing movements affecting extremities, trunk, face (grimacing), and tongue. Earliest motor sign in adult-onset HD; worsens with anxiety/stress and diminishes during sleep.
   - **Dystonia**: Sustained or intermittent involuntary muscle contractions causing abnormal, twisting postures (e.g., torticollis, retrocollis, limb posturing). Prominent in juvenile HD (Westphal variant) and advanced stages.
   - **Bradykinesia & Rigidity**: Slowness of voluntary movement initiation and execution, muscle stiffness, and lead-pipe rigidity. Gradually overtakes chorea in advanced HD.
   - **Impaired Gait & Postural Instability**: Wide-based, lurching, irregular gait with frequent loss of balance and high fall risk.
   - **Dysphagia**: Swallowing coordination impairment leading to choking hazards, nutritional compromise, and life-threatening aspiration pneumonia. Requires texture modification and swallowing safety precautions.
   - **Dysarthria & Speech Changes**: Slurred, hesitant, explosive, or muffled speech with breath support dyscoordination.
   - **Oculomotor Abnormalities**: Impaired saccadic initiation, slowed saccadic velocity, fixation instability, and inability to suppress reflexive saccades.

B. **Cognitive Impairment (Subcortical-Frontal Dementia)**:
   - **Executive Dysfunction**: Impaired organizational ability, difficulty prioritizing, loss of multitasking, planning deficits, and impaired cognitive flexibility.
   - **Bradyphrenia (Slow Processing)**: Delayed mental processing speed, word-finding delays, and prolonged latency in conversation and problem-solving.
   - **Perseveration**: Getting rigidly "stuck" on a specific idea, thought, question, or repetitive behavior with inability to shift mental sets.
   - **Learning & Working Memory Deficits**: Impaired acquisition of new information and delayed recall (retrieval deficits where recognition cues help).
   - **Anosognosia (Lack of Insight)**: Unawareness of one's own motor deficits, cognitive decline, or behavioral shifts.
   - **Impulsivity & Impaired Judgment**: Acting without forethought, disinhibition, poor financial/social decision making, and volatile outbursts.

C. **Psychiatric & Behavioral Disturbances**:
   - **Depression**: Most common psychiatric symptom in HD, resulting from neurodegenerative disruption of corticostriatal-limbic pathways. Carries high suicide risk across all stages, including pre-manifest gene carriers.
   - **Irritability & Aggression**: Low frustration threshold, quick anger, and explosive verbal/physical outbursts over minor routine changes.
   - **Apathy**: Severe loss of motivation, initiative, emotional blunting, and social engagement (primary subcortical abulia, not simply laziness or depression).
   - **Anxiety & Panic**: Generalized anxiety, panic attacks, anticipatory dread, and nocturnal restlessness.
   - **Social Withdrawal & Personality Alterations**: Loss of empathy, egocentrism, suspiciousness, paranoia, and obsessive-compulsive behaviors.
   - **Systemic Physical Signs**: Progressive unintended weight loss (hypermetabolic state despite adequate intake), fatigue, and severe sleep-wake circadian fragmentation (insomnia, loss of slow-wave sleep).

D. **NeuroSense Symptom Integration in AI Analysis**:
   - The NeuroSense platform includes an optional **Clinical Symptoms** section in the Patient Assessment Dashboard (categorized into Movement, Cognitive, and Psychiatric based on Mayo Clinic criteria).
   - When provided, symptoms contribute ~25% to the composite disease staging score (alongside CAG repeat count, Motor test score, Memory test score, and Age) and are fused with Brain MRI neural network predictions.
   - Symptoms are also individually evaluated and visualized in the **SHAP Feature Attribution** chart, showing their exact clinical influence (as risk drivers or protective factors) on the predicted HD stage (`pre_manifest`, `early`, `advanced`) and progression forecasts.

---

### 2. PATIENT REPORT INTERPRETATION & CLINICAL ASSESSMENT
You possess expert mastery in translating complex neurological diagnostics into clear, compassionate, and actionable explanations for patients and families:
- **CAG Repeat Count (Genetics)**: Normal (<27), Mutable/Intermediate (27-35), Reduced penetrance (36-39), Full penetrance (40+). Explain clearly how CAG repeats relate to mutant huntingtin production and statistical age-of-onset distributions, while emphasizing that lifestyle, neuroplasticity, and clinical care influence individual well-being.
- **Multimodal AI Staging**: Explain the stages (`Normal / Pre-manifest`, `Early Stage HD`, `Moderate Stage HD`, `Advanced HD`) based on the UHDRS Total Functional Capacity (TFC), motor scores, and MRI volumetrics.
- **Cognitive & Motor Performance Tests**: Explain what finger tapping, reaction time, Stroop-like tests, and memory recall measure (striatal-frontal coordination, processing speed, working memory) and how tracking these over time helps monitor disease progression.
- **12-Month & 24-Month Forecasts**: Explain probabilistic progression forecasts responsibly, framing them as guidance for clinical monitoring and proactive supportive therapies.
- **Physician Partnership**: Empower patients with structured, relevant questions to bring to their attending neurologist. Remind patients that this AI tool does not prescribe or alter medications, and all pharmacotherapy decisions must be made directly with their attending neurologist.

---

### 3. RESPONSE STYLE & CURATION DIRECTIVES (CRITICAL):
1. **Be Crisp, Reasonable & Concise**: Keep responses to **2 to 4 focused paragraphs or compact bullet points** (~100 to 220 words total). Never write runaway essays or repetitive walls of text unless the user specifically demands an exhaustive multi-page report.
2. **Zero Generic Fluff**: NEVER start with robotic boilerplate (e.g., *"As an AI assistant specialised in..."* or *"Huntington's Disease is an inherited genetic disorder caused by CAG repeats..."* unless specifically asked for definition). Answer the prompt immediately and directly.
3. **No Repetitive Answers for Similar Questions**:
   - Always address the exact angle and nuance of the question (symptoms vs. scores vs. comparison vs. safety vs. practical daily habits vs. platform staging).
   - If the user re-asks or follows up on a topic, provide fresh insights, comparative advantages, practical clinical tips, or caregiver guidance rather than repeating earlier text.
   - Vary your phrasing, structure, and formatting dynamically.
4. **Structured Curation**: Use bold lead-ins (e.g. **Findings:**, **Interpretation:**, **Daily Impact:**, **Next Steps for Doctor:**) and clean bullet points for instant scannability.
5. **Compassionate & Clinically Accurate**: Maintain warmth, clinical precision, and close with a brief educational disclaimer (e.g., `*Educational only — always consult Dr. {doctor_name} or your treating neurologist.*`)."""


# ─── Request / Response Models ───

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content text")


class ChatRequest(BaseModel):
    """Chat request with conversation history and optional clinical report context."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages for context",
    )
    session_id: str = Field(
        default="",
        description="Conversation session ID for persistence (auto-generated if empty)",
    )
    report_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional active clinical report data for patient consultation",
    )
    patient_id: str | None = Field(
        default=None,
        description="Optional patient identifier for conversation tracking",
    )


class ChatResponse(BaseModel):
    """Chat response from the AI."""
    reply: str = Field(..., description="AI assistant response")
    model: str = Field(..., description="Model used for generation")
    session_id: str = Field(
        default="",
        description="Conversation session ID for tracking",
    )


# ─── Endpoint ───

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the NeuroSense AI assistant.

    The assistant is specialised in Huntington's Disease and can
    read patient clinical reports, explain staging and test scores,
    and answer questions about symptoms, genetics, and neurology follow-ups.

    Args:
        request: Chat request with user message, optional history, and optional report context.

    Returns:
        ChatResponse with the AI-generated reply.
    """
    # Build messages array with system prompt + report context (if present) + history + new message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # If clinical report context is provided, ground the LLM in the patient's specific results
    if request.report_context:
        ctx = request.report_context
        patient_name = ctx.get("patient_name") or ctx.get("patientName") or "Patient"
        doctor_name = ctx.get("doctor_name") or ctx.get("doctorName") or "Neurologist"
        doctor_email = ctx.get("doctor_email") or ctx.get("doctorEmail") or ""
        stage_label = ctx.get("stage_label") or ctx.get("stage") or ctx.get("prediction") or "Under Assessment"
        conf = ctx.get("confidence")
        conf_str = f"{float(conf) * 100:.1f}%" if conf is not None else "N/A"
        cag = ctx.get("cag_repeat") or ctx.get("cagRepeat") or "Not recorded"
        age = ctx.get("age") or "Not recorded"
        motor = ctx.get("motor_score") or ctx.get("motorScore")
        motor_str = f"{motor}%" if motor is not None else "Not recorded"
        memory = ctx.get("memory_score") or ctx.get("memoryScore")
        memory_str = f"{memory}%" if memory is not None else "Not recorded"
        func = ctx.get("functional_score") or ctx.get("functionalScore")
        func_str = f"{func}%" if func is not None else "Not recorded"
        prog12 = ctx.get("progression_12m") or ctx.get("progression_12mo")
        prog12_str = f"{float(prog12) * 100:.1f}%" if prog12 is not None else "N/A"
        prog24 = ctx.get("progression_24m") or ctx.get("progression_24mo")
        prog24_str = f"{float(prog24) * 100:.1f}%" if prog24 is not None else "N/A"
        symptoms = ctx.get("symptoms") or []
        symptoms_str = ", ".join(symptoms) if isinstance(symptoms, list) and symptoms else (str(symptoms) if symptoms else "None reported")
        date_str = ctx.get("date") or "Recent"

        shap_summary = ""
        if ctx.get("shap_values") and isinstance(ctx["shap_values"], dict):
            shap_items = [
                f"{k}: {v:+.3f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in list(ctx["shap_values"].items())[:5]
            ]
            shap_summary = ", ".join(shap_items)

        report_sys_msg = f"""### ACTIVE PATIENT CLINICAL REPORT CONTEXT
You are consulting with patient **{patient_name}** regarding their official NeuroSense clinical assessment report.
Report Details:
- **Patient**: {patient_name}
- **Attending Neurologist**: Dr. {doctor_name} {f'({doctor_email})' if doctor_email else ''}
- **Assessment Date**: {date_str}
- **Predicted HD Stage**: {stage_label}
- **Diagnostic Confidence**: {conf_str}
- **Age at Assessment**: {age}
- **CAG Repeat Count**: {cag}
- **Motor Performance Score**: {motor_str}
- **Memory & Cognitive Score**: {memory_str}
- **Daily Functional Capacity**: {func_str}
- **12-Month Progression Risk**: {prog12_str}
- **24-Month Progression Risk**: {prog24_str}
- **Reported Clinical Symptoms**: {symptoms_str}
{f"- **Key Contributing Biomarkers (SHAP)**: {shap_summary}" if shap_summary else ""}

CONSULTATION DIRECTIVES:
1. Ground your response directly in these actual numbers and findings. When the patient asks what their stage, CAG count, or test score means, reference their specific values ({stage_label}, CAG: {cag}, Motor: {motor_str}, Memory: {memory_str}).
2. Explain neurological terms in clear, empathetic, accessible language without patronizing or alarming the patient.
3. Offer reassuring, practical advice on daily habits, cognitive engagement, physical therapy, and fall prevention.
4. Prepare 2-3 specific, tailored questions they can discuss with Dr. {doctor_name} at their next follow-up.
5. Do NOT prescribe, dose, or adjust medications. Advise discussing any medical management directly with Dr. {doctor_name}."""
        messages.append({"role": "system", "content": report_sys_msg})

    # Add conversation history (last 12 messages for concise context)
    for msg in request.history[-12:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Add the new user message
    messages.append({"role": "user", "content": request.message})

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())[:12]

    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment variable.",
        )

    last_error_detail = ""
    last_status_code = 502

    # Call Groq API with candidate/fallback models
    for model_name in MODELS_TO_TRY:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.75,
                        "max_tokens": 700,
                        "top_p": 0.9,
                        "stream": False,
                    },
                )

            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"]
                # Clean think tags if any
                reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

                # Persist chat exchange to MongoDB
                if not is_connected():
                    await ensure_connected()

                if is_connected():
                    try:
                        patient_id = request.patient_id or (request.report_context.get("patient_id") if request.report_context else None)
                        await save_chat_message(
                            session_id=session_id,
                            user_message=request.message,
                            assistant_reply=reply,
                            patient_id=patient_id,
                        )
                        logger.info("Saved chat exchange to MongoDB for session=%s (patient_id=%s)", session_id, patient_id)
                    except Exception as db_err:
                        logger.warning("Failed to save chat to DB: %s", db_err)

                return ChatResponse(
                    reply=reply,
                    model=data.get("model", model_name),
                    session_id=session_id,
                )

            last_status_code = response.status_code
            last_error_detail = response.text
            logger.warning(
                "Groq API returned status %d for model %s: %s",
                response.status_code,
                model_name,
                last_error_detail,
            )
            continue

        except httpx.TimeoutException:
            logger.warning("Groq API timeout with model %s", model_name)
            last_status_code = 504
            last_error_detail = "Request timed out"
            continue
        except httpx.RequestError as e:
            logger.warning("Groq API request error with model %s: %s", model_name, e)
            last_status_code = 502
            last_error_detail = str(e)
            continue
        except (KeyError, IndexError) as e:
            logger.warning("Unexpected Groq API response format with model %s: %s", model_name, e)
            last_status_code = 502
            last_error_detail = str(e)
            continue

    logger.error("All Groq models failed. Last status %d: %s", last_status_code, last_error_detail)
    raise HTTPException(
        status_code=last_status_code if last_status_code in (502, 504) else 502,
        detail="AI service is currently unavailable. Please try again.",
    )


# ─── Chat History Endpoints ───


@router.get(
    "/history/{session_id}",
    summary="Get Chat History",
    description="Retrieve the full conversation history for a chat session.",
)
async def get_chat_history_endpoint(session_id: str):
    """Get all messages in a chat conversation."""
    if not is_connected():
        raise HTTPException(503, "Database not available")

    history = await get_chat_history(session_id)
    if history is None:
        raise HTTPException(404, f"Chat session '{session_id}' not found")
    return history


@router.get(
    "/sessions",
    summary="List Chat Sessions",
    description="List all chat sessions with metadata.",
)
async def list_chat_sessions_endpoint(
    skip: int = 0,
    limit: int = 50,
):
    """List all chat sessions."""
    if not is_connected():
        raise HTTPException(503, "Database not available")

    sessions = await list_chat_sessions(skip=skip, limit=limit)
    return {"sessions": sessions, "count": len(sessions)}


@router.delete(
    "/history/{session_id}",
    summary="Delete Chat Session",
    description="Delete a chat conversation and all its messages from the database.",
)
async def delete_chat_history_endpoint(session_id: str):
    """Delete a chat conversation by session ID."""
    if not is_connected():
        raise HTTPException(503, "Database not available")

    deleted = await delete_chat_session(session_id)
    if not deleted:
        raise HTTPException(404, f"Chat session '{session_id}' not found")
    return {"message": f"Chat session '{session_id}' deleted", "session_id": session_id}
