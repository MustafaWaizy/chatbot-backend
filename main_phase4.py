# main_phase4.py
"""
FastAPI entrypoint for UnityToServe Chatbot (Phase 4)
Integrates:
 - router_phase4 (intent + semantic + fuzzy + CrossEncoder re-rank)
 - logger_phase4 (enhanced logging)
 - rule_responses_phase4 (FAQ source)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import importlib
import os

# Phase 4 modules
from router_phase4 import initialize_router as initialize_router_model, route_message
from logger_phase4 import log_unknown_input, get_unmatched_summary
import rule_responses_phase4  # holds faq_dict and get_response

# ==========================================================
# FastAPI app
# ==========================================================
app = FastAPI(
    title="UnityToServe Chatbot API (Phase 4)",
    description="Hybrid Smart Chatbot Backend (intent + semantic + fuzzy + CrossEncoder, suggestion fallback only)",
    version="4.0"
)

# ==========================================================
# CORS Middleware (local + production + EC2)
# ==========================================================
FRONTEND_URLS = [
    "http://localhost:3000",                         # Local dev frontend
    "https://my-nextjs-site-mu-two.vercel.app",      # Vercel production
    "https://54.162.102.115",                        # EC2 public IP (HTTPS)
    "https://54.162.102.115:8000",                   # EC2 backend direct HTTPS access
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Request Models
# ==========================================================
class MessageRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    user_input: str
    correct_response: str

# ==========================================================
# Startup logic
# ==========================================================
@app.on_event("startup")
async def startup_event():
    """
    Initialize the router semantic model (SentenceTransformer + CrossEncoder)
    at FastAPI startup.
    """
    print("[STARTUP] Initializing Phase 4 services...")
    await asyncio.sleep(0.1)

    try:
        initialize_router_model()
        print("[STARTUP] Router model and CrossEncoder initialized.")
    except Exception as e:
        print(f"[STARTUP WARNING] Router initialization failed: {e}")

    print("[STARTUP] Phase 4 Chatbot ready.")

# ==========================================================
# Routes
# ==========================================================
@app.get("/")
def root():
    return {"status": "UnityToServe Chatbot API (Phase 4) is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "service healthy"}

@app.post("/chat")
async def chat_handler(request: MessageRequest):
    """
    Main chat endpoint: returns structured response dict from router.
    No GPT fallback; suggestion fallback only.
    """
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message provided.")

    # Use hierarchical router (intent + semantic + fuzzy + suggestion fallback)
    result = route_message(user_msg)

    return result

@app.post("/feedback")
async def feedback_handler(feedback: FeedbackRequest):
    """
    Record user/admin feedback to unmatched_inputs log for retraining.
    """
    log_unknown_input(
        user_input=feedback.user_input,
        suggestions=[feedback.correct_response],
        source="user_feedback",
        scores={}
    )
    return {"status": "success", "message": "Feedback recorded."}

@app.post("/reload_faq")
async def reload_faq():
    """
    Reload the FAQ JSON and re-initialize router embeddings (SentenceTransformer + CrossEncoder).
    """
    try:
        importlib.reload(rule_responses_phase4)
        from importlib import reload
        import router_phase4
        reload(router_phase4)

        try:
            router_phase4.initialize_router()
        except Exception:
            initialize_router_model()

        return {
            "status": "success",
            "message": f"Reloaded FAQ ({len(rule_responses_phase4.faq_dict)} entries) and refreshed embeddings."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload FAQ: {e}")

@app.get("/stats")
async def stats(limit: Optional[int] = 10):
    """
    Returns recent unmatched inputs for admin review.
    """
    try:
        recent = get_unmatched_summary(limit)
        return {"count": len(recent), "recent_unmatched": recent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {e}")

# ==========================================================
# HTTPS Notes
# ==========================================================
# ➤ Local Development (no SSL):
#   uvicorn main_phase4:app --host 0.0.0.0 --port 8000
#
# ➤ Local HTTPS (optional testing):
#   uvicorn main_phase4:app --host 0.0.0.0 --port 8000 \
#       --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
#
# ➤ Production on EC2 (recommended with Nginx reverse proxy):
#   sudo systemctl start nginx
#   Nginx handles HTTPS (443) and proxies traffic to FastAPI (port 8000)
#
#   Example proxy block:
#     location / {
#         proxy_pass http://127.0.0.1:8000;
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto $scheme;
#     }
#
# ==========================================================
