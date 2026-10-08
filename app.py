import os
import socket
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from fastapi.responses import JSONResponse
from neurosense.api.main import app as fastapi_app

# ZeroGPU requires at least one @spaces.GPU function during startup
try:
    import spaces

    @spaces.GPU(duration=60)
    def _zero_gpu_worker():
        """Satisfies ZeroGPU startup requirement for Hugging Face Spaces."""
        return True
except ImportError:
    pass

# Root endpoint for FastAPI
@fastapi_app.get("/", tags=["Health"])
async def root_status():
    return JSONResponse({
        "status": "online",
        "service": "NeuroSense Medical AI Backend",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/gradio",
    })

# Status dashboard for Space
with gr.Blocks(title="NeuroSense AI API") as demo:
    gr.Markdown("""
    # 🧠 NeuroSense — Huntington's Disease AI Backend
    The **NeuroSense Medical AI backend** is active and serving API requests.
    
    * 📖 **Swagger API Docs:** [/docs](/docs)
    * 🔍 **ReDoc Docs:** [/redoc](/redoc)
    * 🩺 **Health Check:** [/health](/health)
    * 🔮 **Predictions:** `POST /predict`
    * 💬 **Chatbot:** `POST /chatbot/chat`
    """)

# Mount Gradio onto FastAPI at /gradio so root API routes belong 100% to FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

def get_target_port() -> int:
    """Find available port, respecting ZeroGPU and HF environment variables."""
    if "GRADIO_SERVER_PORT" in os.environ:
        return int(os.environ["GRADIO_SERVER_PORT"])
    if "PORT" in os.environ:
        return int(os.environ["PORT"])
    for p in range(7860, 7875):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return 7860

if __name__ == "__main__":
    import uvicorn
    port = get_target_port()
    print(f"Starting NeuroSense FastAPI server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
