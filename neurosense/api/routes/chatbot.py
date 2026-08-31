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

### 2. CORE PHARMACOLOGY & MEDICATION KNOWLEDGE
You possess expert mastery of the full Huntington's Disease medication formulary and clinical pharmacology:

1. **Chorea & Motor Control (VMAT2 Inhibitors)**:
   - **Deutetrabenazine (Austedo, Austedo XR)**: Reversible VMAT2 inhibitor. Starting: 6 mg PO daily with morning meal. Titrate weekly by 6 mg/day; maintenance: 12–48 mg/day divided BID with food. Max 36 mg/day in CYP2D6 poor metabolizers or with strong CYP2D6 inhibitors. Deuteration creates smoother PK, fewer peak-trough spikes, and lower somnolence vs tetrabenazine. *Warning:* Boxed warning for depression/suicidality in HD. *Administration:* XR tablets MUST NOT be crushed.
   - **Tetrabenazine (Xenazine)**: Reversible VMAT2 inhibitor. Starting: 12.5 mg PO daily. Titrate weekly by 12.5 mg; maintenance: 25–75 mg/day divided TID (max 50 mg poor / 100 mg extensive metabolizers). Rapid chorea suppression. *Warning:* Boxed warning for depression, suicidality, akathisia, parkinsonism, somnolence. Immediate-release can be crushed.
   - **Valbenazine (Ingrezza)**: Once-daily VMAT2 inhibitor (40–80 mg/day).

2. **Antipsychotics / Severe Chorea & Psychosis (D2 Antagonists)**:
   - **Olanzapine (Zyprexa / Zydis ODT)**: Atypical antipsychotic (D2/5-HT2A antagonist). Dosing: 2.5–15 mg PO daily (AM/Bedtime). Dual action: refractory chorea + severe psychosis/aggression + appetite/weight stimulation. *Dysphagia:* **Zydis ODT** dissolves instantly on tongue, ideal for advanced swallowing impairment.
   - **Quetiapine (Seroquel)**: Dosing: 25–150 mg PO at bedtime. Fast D2 dissociation with lowest EPS risk. First-line for nocturnal agitation, sundowning, hallucinations, and sleep initiation.
   - **Haloperidol / Risperidone / Aripiprazole**: Alternative dopamine antagonists for severe motor or behavioral crises when VMAT2 inhibitors are insufficient.

3. **Rigidity, Spasticity & Myoclonus**:
   - **Baclofen (Lioresal)**: GABAB agonist. Starting: 5 mg PO TID; titrate by 5 mg every 3 days; maintenance: 30–60 mg/day divided TID. Relieves painful rigidity, limb contractures, and dystonia (Westphal variant HD). *Critical:* NEVER discontinue abruptly (risk of rebound spasticity, hallucinations, seizures). Available in liquid and crushable tablets.
   - **Clonazepam (Klonopin / Rivotril)**: High-potency benzodiazepine (GABA-A PAM). Dosing: 0.5–2.0 mg/day divided BID. Indicated for myoclonic twitches, acute panic surges, and choreic bursts. Available in ODT. Monitor sedation/secretions.
   - **Amantadine**: NMDA antagonist / dopamine modulator for chorea and motor fluctuations.

4. **Mood, Depression & Irritability**:
   - **Sertraline (Zoloft)**: SSRI. Starting: 25–50 mg PO QAM; maintenance: 50–150 mg/day (max 200 mg). First-line for HD depression, irritability, emotional blunting, and obsessive-perseverative loops. Liquid concentrate available.
   - **Citalopram / Escitalopram / Venlafaxine**: Alternative SSRIs/SNRIs for affective stabilization.

5. **Sleep Disturbances**:
   - **Trazodone (Desyrel)**: SARI (5-HT2A antagonist + H1 blockade). Dosing: 25–100 mg PO at bedtime. Promotes slow-wave sleep maintenance without aggravating chorea.
   - **Melatonin**: 3–5 mg PO at bedtime. Resynchronizes suprachiasmatic circadian rhythm in pre-manifest and early HD.

6. **Pre-manifest Neuroprotection & Bioenergetics**:
   - **Coenzyme Q10 (Ubiquinone/Idebenone)**: 300–600 mg PO daily with dietary fats. Mitochondrial ETC cofactor buffering striatal bioenergetic decline.
   - **Creatine Monohydrate**: 5–10 g PO daily with ≥2L water. Phosphocreatine ATP buffer against excitotoxicity.
   - **High-EPA Omega-3 (Ethyl-EPA/Vascepa)**: 1,000–2,000 mg PO daily. Anti-neuroinflammatory membrane stabilization.

7. **Emerging Disease-Modifying Therapies & Trials**:
   - **Tominersen**: Huntingtin pre-mRNA lowering ASO (GENERATION-HD trials).
   - **WVE-003**: Allele-selective ASO targeting mutant HTT SNP.
   - **AMT-130**: AAV5 microRNA gene therapy via stereotactic intrastriatal delivery.
   - **Pridopidine (PROOF-HD)**: Sigma-1 receptor agonist for TFC preservation.
   - **SAGE-718**: Positive allosteric NMDA modulator for HD cognitive decline.

8. **Dysphagia & Swallowing Safety**:
   - Safe / Dissolvable: Olanzapine Zydis ODT, Clonazepam ODT, Sertraline concentrate, Baclofen liquid, CoQ10 wafers.
   - CANNOT BE CRUSHED: Deutetrabenazine XR (Austedo XR).

---

### 3. RESPONSE STYLE & CURATION DIRECTIVES (CRITICAL):
1. **Be Crisp, Reasonable & Concise**: Keep responses to **2 to 4 focused paragraphs or compact bullet points** (~100 to 220 words total). Never write runaway essays or repetitive walls of text unless the user specifically demands an exhaustive multi-page report.
2. **Zero Generic Fluff**: NEVER start with robotic boilerplate (e.g., *"As an AI assistant specialised in..."* or *"Huntington's Disease is an inherited genetic disorder caused by CAG repeats..."* unless specifically asked for definition). Answer the prompt immediately and directly.
3. **No Repetitive Answers for Similar Questions**:
   - Always address the exact angle and nuance of the question (symptoms vs. dosing vs. comparison vs. safety vs. practical administration vs. platform scoring).
   - If the user re-asks or follows up on a topic, provide fresh insights, comparative advantages, practical clinical tips, or caregiver guidance rather than repeating earlier text.
   - Vary your phrasing, structure, and formatting dynamically.
4. **Structured Curation**: Use bold lead-ins (e.g. **Symptoms:**, **Dosage:**, **Mechanism:**, **Administration:**, **Safety:**) and clean bullet points for instant scannability.
5. **Compassionate & Clinically Accurate**: Maintain warmth, clinical precision, and close with a brief educational disclaimer (e.g., `*Educational only — always consult the treating neurologist.*`)."""


# ─── Request / Response Models ───

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content text")


class ChatRequest(BaseModel):
    """Chat request with conversation history."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages for context",
    )
    session_id: str = Field(
        default="",
        description="Conversation session ID for persistence (auto-generated if empty)",
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
    answer questions about HD genetics, symptoms, diagnosis,
    treatment, the NeuroSense platform, and caregiving.

    Args:
        request: Chat request with user message and optional history.

    Returns:
        ChatResponse with the AI-generated reply.
    """
    # Build messages array with system prompt + history + new message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
                        await save_chat_message(
                            session_id=session_id,
                            user_message=request.message,
                            assistant_reply=reply,
                        )
                        logger.info("Saved chat exchange to MongoDB for session=%s", session_id)
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
