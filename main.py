import os
import secrets
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from database import Base, engine, get_db
from config import CORS_ORIGINS
from services.inference import load_models
from services import inference as _inference
from models import Participant, Recording

from routes.session import router as session_router
from routes.task import router as task_router
from routes.result import router as result_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # ── Startup ──
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Loading ML models...")
    load_models()
    print("Server ready.")
    yield
    # ── Shutdown ──
    print("Shutting down.")


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
_docs_on = os.getenv("ENABLE_DOCS") == "1"

app = FastAPI(
    title="Parkinson's Disease Speech Screening API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_on else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _docs_on else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(session_router, tags=["Session"])
app.include_router(task_router, tags=["Task Processing"])
app.include_router(result_router, tags=["Results"])


@app.get("/")
def root():
    return {
        "name": "Parkinson's Disease Speech Screening API",
        "version": "2.0.0",
        "status": "running",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Parkinson's Speech Screening API is running",
        "models_loaded": {k: len(v) for k, v in _inference._models.items()},
    }


@app.get("/admin/db", response_class=HTMLResponse, tags=["Admin"])
def admin_db_viewer(token: str = "", db: Session = Depends(get_db)):
    """Browser-friendly view of all DB entries. Hidden unless the secret ADMIN_TOKEN is supplied."""
    if not ADMIN_TOKEN or not secrets.compare_digest(token.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=404)
    participants = db.query(Participant).order_by(Participant.created_at.desc()).all()
    recordings = db.query(Recording).order_by(Recording.created_at.desc()).all()

    p_rows = ""
    for p in participants:
        pred = {0: "Healthy", 1: "Parkinson's"}.get(p.final_prediction, "—")
        conf = f"{p.final_confidence:.4f}" if p.final_confidence is not None else "—"
        p_rows += f"<tr><td>{p.id}</td><td><code>{p.participant_id[:12]}…</code></td><td>{pred}</td><td>{conf}</td><td>{p.created_at:%Y-%m-%d %H:%M:%S}</td></tr>"

    r_rows = ""
    for r in recordings:
        label = {0: "✅ Healthy", 1: "🔴 Parkinson's"}.get(r.label, "—")
        conf = f"{r.confidence:.4f}" if r.confidence is not None else "—"
        r_rows += f"<tr><td>{r.id}</td><td><code>{r.participant_id[:12]}…</code></td><td>{r.task_id}</td><td>{r.task_name}</td><td>{label}</td><td>{conf}</td><td>{r.created_at:%Y-%m-%d %H:%M:%S}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><title>PD Admin — DB Viewer</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ color: #38bdf8; }} h2 {{ color: #7dd3fc; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  th, td {{ padding: 8px 12px; border: 1px solid #334155; text-align: left; }}
  th {{ background: #1e293b; color: #38bdf8; }}
  tr:nth-child(even) {{ background: #1e293b; }}
  code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
  .stats {{ display: flex; gap: 2rem; margin: 1rem 0; }}
  .stat {{ background: #1e293b; padding: 1rem 1.5rem; border-radius: 8px; }}
  .stat-num {{ font-size: 1.5rem; font-weight: bold; color: #38bdf8; }}
  a {{ color: #38bdf8; }}
</style></head><body>
<h1>🧠 Parkinson's Screening — DB Viewer</h1>
<div class="stats">
  <div class="stat"><div class="stat-num">{len(participants)}</div>Sessions</div>
  <div class="stat"><div class="stat-num">{len(recordings)}</div>Recordings</div>
  <div class="stat"><div class="stat-num">{sum(1 for r in recordings if r.label==0)}</div>Healthy</div>
  <div class="stat"><div class="stat-num">{sum(1 for r in recordings if r.label==1)}</div>Parkinson's</div>
</div>
<p><a href="/admin/db?token={token}">🔄 Refresh</a></p>

<h2>📋 Recordings (model predictions)</h2>
<table>
<tr><th>ID</th><th>Participant</th><th>Task</th><th>Task Name</th><th>Prediction</th><th>Confidence</th><th>Time</th></tr>
{r_rows if r_rows else "<tr><td colspan='7' style='text-align:center'>No recordings yet</td></tr>"}
</table>

<h2>👥 Participants (sessions)</h2>
<table>
<tr><th>ID</th><th>Participant</th><th>Final Prediction</th><th>Confidence</th><th>Created</th></tr>
{p_rows if p_rows else "<tr><td colspan='5' style='text-align:center'>No sessions yet</td></tr>"}
</table>
</body></html>"""
    return HTMLResponse(content=html)
