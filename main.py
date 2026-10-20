"""
main.py

AI Interview Platform
Authentication + Resume Analysis + Interview WebSocket
"""

import uuid
import json
import logging
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    Header,
    HTTPException,
    Query,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from pydantic import BaseModel

from services import (
    resume_service,
    llm_service,
    speech_service,
    vision_service,
)

from services.interview_service import (
    InterviewOrchestrator
)

import database
import auth_service


# ============================================================
# APP
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("interview")

app = FastAPI(
    title="AI Interview Platform"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# SESSIONS
# ============================================================

SESSIONS: dict[str, InterviewOrchestrator] = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):

    username: str
    email: str
    password: str


class LoginRequest(BaseModel):

    email: str
    password: str


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def on_startup():

    database.init_database()

    try:

        await speech_service.ensure_model_loaded()

    except Exception as e:

        logger.error(
            f"faster-whisper failed to load at startup: {e}"
        )

    if vision_service.is_available():

        logger.info(
            "MediaPipe vision analysis is available."
        )

    else:

        logger.warning(
            f"Vision analysis disabled: "
            f"{vision_service.unavailable_reason()}"
        )


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Interview Platform is running"
    }


@app.get("/app")
def app_page():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


@app.get("/login")
def login_page():

    # There is no separate login.html — the login screen is the
    # #authScreen block inside index.html itself, so this route
    # serves the same single-page file as /app. Previously this
    # pointed at "login.html", which does not exist, and every
    # GET /login crashed with a 500 (FileNotFoundError / Starlette
    # RuntimeError: File at path ... does not exist).

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ============================================================
# AUTHENTICATION
# ============================================================

@app.post("/register")
def register(request: RegisterRequest):

    username = request.username.strip()

    email = request.email.strip().lower()

    password = request.password

    # -----------------------------
    # Validation
    # -----------------------------

    if len(username) < 3:

        raise HTTPException(
            status_code=400,
            detail="Username must contain at least 3 characters."
        )

    if len(username) > 30:

        raise HTTPException(
            status_code=400,
            detail="Username is too long."
        )

    if "@" not in email or "." not in email:

        raise HTTPException(
            status_code=400,
            detail="Enter a valid email address."
        )

    if len(password) < 6:

        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters."
        )

    # -----------------------------
    # Check existing user
    # -----------------------------

    existing_user = database.get_user_by_email(email)

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists."
        )

    # -----------------------------
    # Hash password
    # -----------------------------

    password_hash = auth_service.hash_password(
        password
    )

    # -----------------------------
    # Create user
    # -----------------------------

    user = database.create_user(
        username=username,
        email=email,
        password_hash=password_hash
    )

    if user is None:

        raise HTTPException(
            status_code=400,
            detail="Username or email already exists."
        )

    return {
        "success": True,
        "message": "Account created successfully.",
        "user": user
    }


@app.post("/login")
def login(request: LoginRequest):

    email = request.email.strip().lower()

    password = request.password

    user = database.get_user_by_email(
        email
    )

    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    password_valid = auth_service.verify_password(
        password,
        user["password_hash"]
    )

    if not password_valid:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    token = auth_service.create_access_token(
        user_id=user["id"],
        email=user["email"]
    )

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        }
    }


# ============================================================
# AUTH HELPER
# ============================================================

def get_authenticated_user(
    authorization: str | None
):

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    if not authorization.lower().startswith(
        "bearer "
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header."
        )

    token = authorization.split(
        " ",
        1
    )[1].strip()

    decoded = auth_service.decode_access_token(
        token
    )

    if decoded is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token."
        )

    user = database.get_user_by_id(
        decoded["user_id"]
    )

    if user is None:

        raise HTTPException(
            status_code=401,
            detail="User no longer exists."
        )

    return user


# ============================================================
# CURRENT USER
# ============================================================

@app.get("/me")
def get_current_user(
    authorization: str | None = Header(
        default=None
    )
):

    user = get_authenticated_user(
        authorization
    )

    return {
        "success": True,
        "user": user
    }


# ============================================================
# RESUME → QUESTIONS
# ============================================================

@app.post("/generate-questions")
async def generate_questions(
    file: UploadFile = File(...),
    authorization: str | None = Header(
        default=None
    )
):

    # Authentication
    user = get_authenticated_user(
        authorization
    )

    contents = await file.read()

    try:

        resume_text = (
            resume_service.extract_resume_text(
                contents
            )
        )

    except resume_service.ResumeExtractionError as e:

        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )

    resume_summary = await llm_service.analyze_resume(
        resume_text
    )

    session_id = uuid.uuid4().hex

    orchestrator = InterviewOrchestrator(
        session_id,
        resume_text,
        resume_summary
    )

    # Attach user ID to session
    orchestrator.user_id = user["id"]

    orchestrator.username = user["username"]

    SESSIONS[session_id] = orchestrator

    return {
        "filename": file.filename,
        "session_id": session_id,
        "resume_summary": resume_summary,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        }
    }


# ============================================================
# WEBSOCKET AUTHENTICATION
# ============================================================

@app.websocket("/ws/interview/{session_id}")
async def interview_ws(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(default="")
):

    # --------------------------------------------------------
    # Validate token BEFORE accepting WebSocket
    # --------------------------------------------------------

    decoded = auth_service.decode_access_token(
        token
    )

    if decoded is None:

        await websocket.close(
            code=1008
        )

        return

    # --------------------------------------------------------
    # Find session
    # --------------------------------------------------------

    orchestrator = SESSIONS.get(
        session_id
    )

    if orchestrator is None:

        await websocket.close(
            code=1008
        )

        return

    # --------------------------------------------------------
    # Ensure session belongs to logged-in user
    # --------------------------------------------------------

    if getattr(
        orchestrator,
        "user_id",
        None
    ) != decoded["user_id"]:

        await websocket.close(
            code=1008
        )

        return

    # --------------------------------------------------------
    # Accept
    # --------------------------------------------------------

    await websocket.accept()

    logger.info(
        f"[{session_id}] "
        f"Authenticated WebSocket connected "
        f"user={decoded['user_id']}"
    )

    try:

        while True:

            message = await websocket.receive()

            # =================================================
            # DISCONNECT
            # =================================================

            if message["type"] == "websocket.disconnect":

                break

            # =================================================
            # TEXT
            # =================================================

            if message.get("text") is not None:

                try:

                    msg = json.loads(
                        message["text"]
                    )

                except json.JSONDecodeError:

                    continue

                event = msg.get(
                    "event"
                )

                # ---------------------------------------------
                # START
                # ---------------------------------------------

                if event == "start_interview":

                    result = await orchestrator.start()

                    await websocket.send_json(
                        result
                    )

                # ---------------------------------------------
                # AI DONE SPEAKING
                # ---------------------------------------------

                elif event == "ai_done_speaking":

                    result = (
                        orchestrator
                        .ai_finished_speaking()
                    )

                    await websocket.send_json(
                        result
                    )

                # ---------------------------------------------
                # END
                # ---------------------------------------------

                elif event == "end_interview":

                    result = (
                        await orchestrator
                        .end_interview()
                    )

                    await websocket.send_json(
                        result
                    )

                # ---------------------------------------------
                # UNKNOWN
                # ---------------------------------------------

                else:

                    await websocket.send_json({
                        "event": "error",
                        "message":
                            f"Unknown event: {event}"
                    })

            # =================================================
            # BINARY
            # =================================================

            elif message.get("bytes") is not None:

                payload: bytes = message["bytes"]

                if len(payload) < 1:

                    continue

                tag = payload[0]

                body = payload[1:]

                # ---------------------------------------------
                # AUDIO
                # ---------------------------------------------

                if tag == 1:

                    events = (
                        await orchestrator
                        .handle_audio_chunk(
                            body
                        )
                    )

                    for event in events:

                        await websocket.send_json(
                            event
                        )

                # ---------------------------------------------
                # VIDEO
                # ---------------------------------------------

                elif tag == 2:

                    event = (
                        await orchestrator
                        .handle_video_frame(
                            body
                        )
                    )

                    if event:

                        await websocket.send_json(
                            event
                        )

    except WebSocketDisconnect:

        logger.info(
            f"[{session_id}] "
            f"WebSocket disconnected"
        )

    except Exception as e:

        logger.exception(
            f"[{session_id}] "
            f"WebSocket error: {e}"
        )