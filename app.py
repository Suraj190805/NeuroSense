import os
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from neurosense.api.main import app as fastapi_app

# Informative status interface for the Hugging Face Space
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

# Mount Gradio onto FastAPI so both the status page and all FastAPI endpoints work
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
