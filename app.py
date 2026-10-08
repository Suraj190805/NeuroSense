import os
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Status dashboard for Space root
with gr.Blocks(title="NeuroSense AI API") as demo:
    gr.Markdown("""
    # 🧠 NeuroSense — Huntington's Disease AI Backend
    
    The **NeuroSense Medical AI backend** is active and serving API requests.
    
    ### 🔗 Available Endpoints:
    * 📖 **Swagger API Docs:** [/docs](/docs)
    * 🔍 **ReDoc Docs:** [/redoc](/redoc)
    * 🩺 **Health Check:** [/health](/health)
    * 🔮 **Predictions:** `POST /predict`
    * 💬 **Chatbot:** `POST /chatbot/chat`
    
    ---
    *Backend service for the NeuroSense Vercel web application.*
    """)

# Prepend all FastAPI routes so they take priority over Gradio's SvelteKit handlers
demo.app.router.routes = list(fastapi_app.router.routes) + demo.app.router.routes

# Mount static heatmaps directory
heatmap_dir = Path("outputs/heatmaps")
heatmap_dir.mkdir(parents=True, exist_ok=True)
demo.app.mount("/static/heatmaps", StaticFiles(directory=str(heatmap_dir)), name="heatmaps")

# Ensure CORS allows Vercel frontend requests
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # Gradio launch handles ZeroGPU internal port allocation automatically
    demo.launch(server_name="0.0.0.0")
