import datetime
import os
import re
import base64
import secrets
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field

from app.database import init_db, get_db, DotDict, get_next_id, assign_doctor_round_robin
from app.auth import hash_password, verify_password, create_access_token, get_current_user, require_role
from app.model_service import predict_distress, wants_doctor, personalize_recommendation, answer_about_me, answer_app_faq, get_recommendation

ALL_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")  # HH:MM, 24-hour

app = FastAPI(title="PPD Support Platform API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

if os.environ.get("PPD_APP_SECRET") is None:
    print(
        "\n*** WARNING: PPD_APP_SECRET is not set -- using the insecure dev default. "
        "Anyone who reads this source code can forge login tokens for any user. "
        "Set a real random PPD_APP_SECRET environment variable before deploying "
        "anywhere other than your own machine. ***\n"
    )

# Doctor verification documents are stored as base64 in the doctor's own
# Mongo document (well under Mongo's 16MB doc limit for a <=10MB file),
# rather than on local disk. This matters for deployment: free-tier hosts
# like Render/Railway have EPHEMERAL filesystems -- anything written to
# local disk is silently wiped on every restart or redeploy, which would
# have made every doctor's uploaded ID vanish. Storing it in the same place
# as the rest of the user's data means it survives exactly as long as the
# account does, with no separate persistent-volume setup needed.
ALLOWED_DOC_EXTENSIONS = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAX_DOC_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=6)
    role: str = "mother"  # mother | doctor  (admin created manually / seeded)
    specialty: Optional[str] = None
    # Doctor-only profile fields, shown to mothers when booking a physical
    # appointment (e.g. "Dr. Mahesh Mehta, MD Psychiatry, Sunrise Clinic,
    # Mumbai - Panvel"). All optional at signup, editable later via
    # PUT /doctor/profile.
    qualification: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    # Days/hours the doctor is generally bookable -- used to constrain the
    # date/time picker on the booking form. Defaults to every day, 9-5.
    available_days: Optional[List[str]] = None  # e.g. ["Mon","Tue","Wed","Thu","Fri"]
    available_start_time: Optional[str] = "09:00"
    available_end_time: Optional[str] = "17:00"
    # Mother-only personalization fields -- used to tailor stress-reduction
    # suggestions (e.g. nudging toward a hobby the mother already enjoys).
    # Never used for anything safety-critical: distress classification and
    # doctor referrals never read these fields, only the recommendation text does.
    hobbies: Optional[str] = None
    background_info: Optional[str] = None
    # Optional -- how the mother prefers the bot's non-critical replies to
    # sound. Never affects classification or the Emotionally Distressed
    # safety-referral message, which always stays serious regardless of this.
    tone_preference: Optional[str] = "warm"  # "warm" | "practical" | "brief"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    label: int
    label_name: str
    confidence: float
    explanation: list
    recommendation: str
    needs_doctor: bool
    show_doctors: bool = False
    doctors: list = []


class BookAppointmentRequest(BaseModel):
    doctor_id: int
    mode: str  # "online" | "offline"
    appointment_date: str  # "YYYY-MM-DD"
    appointment_time: str  # "HH:MM", 24-hour
    notes: Optional[str] = None


class PostRequest(BaseModel):
    text: str


class ReplyRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
VALID_TONES = {"warm", "practical", "brief"}


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db=Depends(get_db)):
    if payload.role not in ("mother", "doctor"):
        raise HTTPException(400, "role must be 'mother' or 'doctor' at signup")
    if payload.role == "mother":
        if not payload.hobbies or not payload.hobbies.strip():
            raise HTTPException(400, "Hobbies / interests is required so we can personalize your support")
        if not payload.background_info or not payload.background_info.strip():
            raise HTTPException(400, "A little about you is required so we can personalize your support")
        if payload.tone_preference and payload.tone_preference not in VALID_TONES:
            raise HTTPException(400, f"tone_preference must be one of {sorted(VALID_TONES)}")
    if payload.role == "doctor":
        for label, val in (("available_start_time", payload.available_start_time),
                            ("available_end_time", payload.available_end_time)):
            if val and not TIME_RE.match(val):
                raise HTTPException(400, f"{label} must be in HH:MM 24-hour format")
        if payload.available_days:
            bad = [d for d in payload.available_days if d not in ALL_WEEKDAYS]
            if bad:
                raise HTTPException(400, f"available_days contains invalid values: {bad}. Must be from {ALL_WEEKDAYS}")
    existing = db["users"].find_one({"email": payload.email})
    if existing:
        raise HTTPException(400, "Email already registered")

    user_doc = {
        "id": get_next_id("users"),
        "name": payload.name,
        "email": payload.email,
        "hashed_password": hash_password(payload.password),
        "role": payload.role,
        "specialty": payload.specialty if payload.role == "doctor" else None,
        "qualification": payload.qualification if payload.role == "doctor" else None,
        "clinic_name": payload.clinic_name if payload.role == "doctor" else None,
        "clinic_address": payload.clinic_address if payload.role == "doctor" else None,
        "available_days": (payload.available_days or list(ALL_WEEKDAYS)) if payload.role == "doctor" else None,
        "available_start_time": (payload.available_start_time or "09:00") if payload.role == "doctor" else None,
        "available_end_time": (payload.available_end_time or "17:00") if payload.role == "doctor" else None,
        "hobbies": payload.hobbies if payload.role == "mother" else None,
        "background_info": payload.background_info if payload.role == "mother" else None,
        "tone_preference": (payload.tone_preference or "warm") if payload.role == "mother" else None,
        # Doctors start unverified -- they must upload a credential document
        # (POST /doctor/verification/document) and be approved by an admin
        # before they appear to mothers or receive referrals. See
        # assign_doctor_round_robin() and list_doctors() for the gating.
        "verification_status": "unverified" if payload.role == "doctor" else None,
        "verification_doc_base64": None,
        "verification_doc_original_name": None,
        "verification_note": None,
        # Doctors default to offline -- they explicitly go online when
        # they're actually available, rather than appearing available the
        # moment they're verified. See PUT /doctor/availability.
        "availability_status": "offline" if payload.role == "doctor" else None,
        "created_at": datetime.datetime.utcnow(),
    }
    db["users"].insert_one(user_doc)
    user = DotDict(user_doc)

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token, role=user.role, name=user.name)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db=Depends(get_db)):
    user_doc = db["users"].find_one({"email": payload.email})
    if not user_doc or not verify_password(payload.password, user_doc["hashed_password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    user = DotDict(user_doc)

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token, role=user.role, name=user.name)


@app.get("/auth/me")
def me(user: DotDict = Depends(get_current_user)):
    out = {"id": user.id, "name": user.name, "email": user.email, "role": user.role}
    if user.role == "mother":
        out["hobbies"] = user.hobbies
        out["background_info"] = user.background_info
        out["tone_preference"] = user.tone_preference or "warm"
    if user.role == "doctor":
        out["specialty"] = user.specialty
        out["verification_status"] = user.verification_status or "unverified"
        out["verification_note"] = user.verification_note
        out["availability_status"] = user.availability_status or "offline"
        out["qualification"] = user.qualification
        out["clinic_name"] = user.clinic_name
        out["clinic_address"] = user.clinic_address
        out["available_days"] = user.available_days or list(ALL_WEEKDAYS)
        out["available_start_time"] = user.available_start_time or "09:00"
        out["available_end_time"] = user.available_end_time or "17:00"
    return out


class UpdateProfileRequest(BaseModel):
    hobbies: Optional[str] = None
    background_info: Optional[str] = None
    tone_preference: Optional[str] = None


@app.put("/auth/me/profile")
def update_profile(payload: UpdateProfileRequest, user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    # These are required at signup, so this endpoint only ever supports
    # updating them to something new -- not clearing them, since the model
    # relies on them for every reply.
    update = {}
    if payload.hobbies is not None:
        if not payload.hobbies.strip():
            raise HTTPException(400, "Hobbies / interests cannot be blank")
        update["hobbies"] = payload.hobbies
    if payload.background_info is not None:
        if not payload.background_info.strip():
            raise HTTPException(400, "Background info cannot be blank")
        update["background_info"] = payload.background_info
    if payload.tone_preference is not None:
        if payload.tone_preference not in VALID_TONES:
            raise HTTPException(400, f"tone_preference must be one of {sorted(VALID_TONES)}")
        update["tone_preference"] = payload.tone_preference
    if update:
        db["users"].update_one({"id": user.id}, {"$set": update})
    return {
        "status": "ok",
        "hobbies": payload.hobbies,
        "background_info": payload.background_info,
        "tone_preference": payload.tone_preference,
    }


# ---------------------------------------------------------------------------
# Chat endpoint (mother <-> bot, private, explainable distress detection)
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    # For genuinely ambiguous short replies ("yes", "idk", "a bit better"),
    # the message alone often isn't enough to classify well -- prepend the
    # mother's last couple of messages as context. This is intentionally
    # narrow: it only fires for a curated set of filler-style short replies,
    # NOT any message under the word threshold. A short but substantive
    # disclosure ("my husband beats me everyday" is only 5 words) must be
    # classified on its own, unpolluted by older context -- testing caught a
    # real case where context-padding let a stale phrase from two turns
    # earlier suppress classification of an unrelated, serious new message.
    AMBIGUOUS_SHORT_REPLIES = {
        "yes", "no", "yeah", "yep", "nope", "maybe", "idk", "i dont know",
        "i don't know", "not sure", "a bit", "a little", "better", "worse",
        "same", "okay", "ok", "fine", "meh", "so so", "so-so", "a bit better",
        "a bit worse", "better i guess", "worse i guess", "kind of", "sort of",
    }
    classification_text = payload.text
    if payload.text.strip(" .!?").lower() in AMBIGUOUS_SHORT_REPLIES:
        recent_mother_msgs = list(
            db["chat_messages"]
            .find({"mother_id": user.id, "sender": "mother"})
            .sort("created_at", -1)
            .limit(2)
        )
        if recent_mother_msgs:
            context = " ".join(m["text"] for m in reversed(recent_mother_msgs))
            classification_text = f"{context} . {payload.text}"

    # Fetch this before generating the reply so get_recommendation() can
    # avoid repeating the exact same text as last time, and so we know
    # whether the bot's last message offered a doctor referral (needed below
    # so a mother replying "yes" to that offer is understood, instead of
    # "yes" being scored as an unrelated new message).
    last_bot_msg = db["chat_messages"].find_one(
        {"mother_id": user.id, "sender": "bot"}, sort=[("created_at", -1)]
    )
    previous_bot_text = last_bot_msg["text"] if last_bot_msg else None
    previously_offered = bool(last_bot_msg and last_bot_msg.get("offered_referral"))

    result = predict_distress(classification_text, raw_text=payload.text)
    if result.get("meta_topic") == "app_faq":
        result["recommendation"] = answer_app_faq(previous_text=previous_bot_text)
    elif result.get("meta_topic"):
        result["recommendation"] = answer_about_me(
            result["meta_topic"], user.hobbies, user.background_info
        )
    else:
        base_text = get_recommendation(
            result["label"], tone=user.tone_preference, previous_text=previous_bot_text
        )
        result["recommendation"] = personalize_recommendation(
            result["label"], base_text, user.hobbies
        )

    # Explicit doctor request ("connect me to a doctor", or "yes" right after
    # the bot offered one) is handled separately from the distress
    # classification -- a mother can ask for a doctor at any distress level,
    # not only when the classifier flags her as severe.
    doctor_intent = wants_doctor(payload.text, previous_bot_offered_referral=previously_offered)
    doctors_list = []
    if doctor_intent:
        doctors = list(db["users"].find({"role": "doctor", "verification_status": "verified"}))
        # Online doctors first, so a mother reaching out sees who's actually
        # available to respond sooner rather than scrolling for it.
        doctors.sort(key=lambda d: d.get("availability_status") != "online")
        doctors_list = [_doctor_public_fields(d) for d in doctors]
        if doctors_list:
            result["recommendation"] = (
                "Of course. Here are the doctors available on Amara -- you can request "
                "an online or in-person appointment with whichever one feels right."
            )
        else:
            result["recommendation"] = (
                "I'd like to connect you with a doctor, but none are verified on the "
                "platform yet. Please check back soon."
            )

    now = datetime.datetime.utcnow()
    db["chat_messages"].insert_one({
        "id": get_next_id("chat_messages"),
        "mother_id": user.id, "sender": "mother", "text": payload.text,
        "label": None, "label_name": None, "confidence": None, "explanation": None,
        "offered_referral": False, "created_at": now,
    })
    db["chat_messages"].insert_one({
        "id": get_next_id("chat_messages"),
        "mother_id": user.id, "sender": "bot", "text": result["recommendation"],
        "label": result["label"], "label_name": result["label_name"],
        "confidence": result["confidence"], "explanation": result["explanation"],
        "offered_referral": bool(result["needs_doctor"] or doctor_intent),
        "created_at": now + datetime.timedelta(microseconds=1),  # keep strict ordering after the mother's message
    })

    if result["needs_doctor"]:
        db["doctor_referrals"].insert_one({
            "id": get_next_id("doctor_referrals"),
            "mother_id": user.id, "doctor_id": assign_doctor_round_robin(), "status": "pending",
            "trigger_message": payload.text, "created_at": datetime.datetime.utcnow(),
        })

    result.pop("meta_topic", None)
    return ChatResponse(**result, show_doctors=doctor_intent and bool(doctors_list), doctors=doctors_list)


@app.get("/chat/history")
def chat_history(user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    # Private: only this mother's own messages are ever returned - enforced by mother_id filter
    msgs = list(db["chat_messages"].find({"mother_id": user.id}).sort("created_at", 1))
    return [
        {
            "sender": m["sender"],
            "text": m["text"],
            "label_name": m.get("label_name"),
            "confidence": m.get("confidence"),
            "explanation": m.get("explanation"),
            "created_at": m["created_at"].isoformat(),
        }
        for m in msgs
    ]


# ---------------------------------------------------------------------------
# Community endpoints (visible to all mothers; separate from private chat)
# ---------------------------------------------------------------------------
@app.post("/community/posts")
def create_post(payload: PostRequest, user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    post_doc = {
        "id": get_next_id("community_posts"),
        "author_id": user.id, "text": payload.text, "created_at": datetime.datetime.utcnow(),
    }
    db["community_posts"].insert_one(post_doc)
    return {"id": post_doc["id"], "author": user.name, "text": post_doc["text"], "created_at": post_doc["created_at"].isoformat()}


@app.get("/community/posts")
def list_posts(user: DotDict = Depends(require_role("mother", "admin")), db=Depends(get_db)):
    posts = list(db["community_posts"].find().sort("created_at", -1))
    out = []
    for p in posts:
        author = db["users"].find_one({"id": p["author_id"]})
        replies = list(db["community_replies"].find({"post_id": p["id"]}).sort("created_at", 1))
        out.append({
            "id": p["id"],
            "author": author["name"] if author else "Unknown",
            "text": p["text"],
            "created_at": p["created_at"].isoformat(),
            "replies": [
                {
                    "author": (db["users"].find_one({"id": r["author_id"]}) or {"name": "Unknown"})["name"],
                    "text": r["text"],
                    "created_at": r["created_at"].isoformat(),
                }
                for r in replies
            ],
        })
    return out


@app.post("/community/posts/{post_id}/replies")
def reply_to_post(post_id: int, payload: ReplyRequest, user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    post = db["community_posts"].find_one({"id": post_id})
    if not post:
        raise HTTPException(404, "Post not found")
    db["community_replies"].insert_one({
        "id": get_next_id("community_replies"),
        "post_id": post_id, "author_id": user.id, "text": payload.text,
        "created_at": datetime.datetime.utcnow(),
    })
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Doctor verification
# ---------------------------------------------------------------------------
@app.post("/doctor/verification/document")
async def upload_verification_document(
    file: UploadFile = File(...),
    user: DotDict = Depends(require_role("doctor")),
    db=Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(400, f"File type must be one of {sorted(ALLOWED_DOC_EXTENSIONS)}")

    contents = await file.read()
    if len(contents) > MAX_DOC_SIZE_BYTES:
        raise HTTPException(400, "File is too large (max 10 MB)")
    if not contents:
        raise HTTPException(400, "File is empty")

    db["users"].update_one(
        {"id": user.id},
        {"$set": {
            "verification_status": "pending",
            "verification_doc_base64": base64.b64encode(contents).decode("ascii"),
            "verification_doc_content_type": ALLOWED_DOC_EXTENSIONS[ext],
            "verification_doc_original_name": file.filename,
            "verification_note": None,
        }},
    )
    return {"status": "pending", "message": "Document received -- an admin will review it shortly."}


@app.get("/doctor/verification/status")
def verification_status(user: DotDict = Depends(require_role("doctor"))):
    return {
        "verification_status": user.verification_status or "unverified",
        "verification_note": user.verification_note,
        "has_document": bool(user.verification_doc_base64),
    }


class AvailabilityRequest(BaseModel):
    status: str  # "online" | "offline"


@app.put("/doctor/availability")
def set_availability(payload: AvailabilityRequest, user: DotDict = Depends(require_role("doctor")), db=Depends(get_db)):
    if payload.status not in ("online", "offline"):
        raise HTTPException(400, "status must be 'online' or 'offline'")
    db["users"].update_one({"id": user.id}, {"$set": {"availability_status": payload.status}})
    return {"availability_status": payload.status}


class DoctorProfileRequest(BaseModel):
    qualification: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    available_days: Optional[List[str]] = None
    available_start_time: Optional[str] = None
    available_end_time: Optional[str] = None


@app.put("/doctor/profile")
def update_doctor_profile(payload: DoctorProfileRequest, user: DotDict = Depends(require_role("doctor")), db=Depends(get_db)):
    update = {}
    for field in ("qualification", "clinic_name", "clinic_address"):
        val = getattr(payload, field)
        if val is not None:
            update[field] = val
    for field in ("available_start_time", "available_end_time"):
        val = getattr(payload, field)
        if val is not None:
            if not TIME_RE.match(val):
                raise HTTPException(400, f"{field} must be in HH:MM 24-hour format")
            update[field] = val
    if payload.available_days is not None:
        bad = [d for d in payload.available_days if d not in ALL_WEEKDAYS]
        if bad:
            raise HTTPException(400, f"available_days contains invalid values: {bad}. Must be from {ALL_WEEKDAYS}")
        update["available_days"] = payload.available_days
    if update:
        db["users"].update_one({"id": user.id}, {"$set": update})
    return {"status": "ok", **update}


# ---------------------------------------------------------------------------
# Doctor endpoints
# ---------------------------------------------------------------------------
@app.get("/doctor/referrals")
def list_referrals(user: DotDict = Depends(require_role("doctor", "admin")), db=Depends(get_db)):
    if user.role == "doctor" and user.verification_status != "verified":
        raise HTTPException(403, "Your account must be verified before you can view referrals")

    # Doctors see cases assigned to them AND currently-unclaimed cases.
    # "Unclaimed" happens when a referral was created before assign_doctor_
    # round_robin() had any (verified) doctor to hand it to -- previously
    # this stored doctor_id=None permanently, and since this query used to
    # be an exact match on doctor_id, that referral became invisible to
    # every doctor forever (only admins could see it via the {} query).
    # That was the reported bug: a doctor's referral section could sit
    # empty even with real pending cases in the system. Unclaimed referrals
    # are now visible to every VERIFIED doctor and can be accepted by
    # whoever gets to them first -- see accept_referral() below.
    if user.role == "admin":
        query = {}
    else:
        query = {"$or": [{"doctor_id": user.id}, {"doctor_id": None}]}
    referrals = list(db["doctor_referrals"].find(query).sort("created_at", -1))
    out = []
    for r in referrals:
        mother = db["users"].find_one({"id": r["mother_id"]})
        out.append({
            "id": r["id"],
            "mother_name": mother["name"] if mother else "Unknown",
            "status": r["status"],
            "trigger_message": r.get("trigger_message"),
            "created_at": r["created_at"].isoformat(),
            "assigned_to_you": r.get("doctor_id") == user.id,
            "unclaimed": r.get("doctor_id") is None,
        })
    return out


@app.post("/doctor/referrals/{referral_id}/accept")
def accept_referral(referral_id: int, user: DotDict = Depends(require_role("doctor")), db=Depends(get_db)):
    if user.verification_status != "verified":
        raise HTTPException(403, "Your account must be verified before you can accept referrals")

    # A doctor can accept a case that's either already assigned to them, or
    # currently unclaimed (doctor_id None) -- accepting an unclaimed one
    # claims it for that doctor. Ownership is still enforced server-side:
    # a doctor can never touch a case that's assigned to a DIFFERENT doctor.
    referral = db["doctor_referrals"].find_one({
        "id": referral_id,
        "$or": [{"doctor_id": user.id}, {"doctor_id": None}],
    })
    if not referral:
        raise HTTPException(404, "Referral not found")
    db["doctor_referrals"].update_one(
        {"id": referral_id}, {"$set": {"status": "accepted", "doctor_id": user.id}}
    )
    return {"status": "accepted"}


def _doctor_public_fields(d: dict) -> dict:
    """Fields a mother needs to choose and book a doctor -- shared between
    /doctors and the chat's doctor-connect flow so both stay in sync."""
    return {
        "id": d["id"], "name": d["name"], "specialty": d.get("specialty"),
        "availability_status": d.get("availability_status") or "offline",
        "qualification": d.get("qualification"),
        "clinic_name": d.get("clinic_name"),
        "clinic_address": d.get("clinic_address"),
        "available_days": d.get("available_days") or list(ALL_WEEKDAYS),
        "available_start_time": d.get("available_start_time") or "09:00",
        "available_end_time": d.get("available_end_time") or "17:00",
    }


@app.get("/doctors")
def list_doctors(user: DotDict = Depends(require_role("mother", "admin")), db=Depends(get_db)):
    # Mothers should only ever be connected to a doctor whose credentials
    # have actually been reviewed and approved -- that's the whole point of
    # requiring verification documents. Admins see everyone, including
    # unverified/pending, since they're the ones doing the reviewing.
    query = {"role": "doctor"} if user.role == "admin" else {"role": "doctor", "verification_status": "verified"}
    doctors = list(db["users"].find(query))
    if user.role != "admin":
        doctors.sort(key=lambda d: d.get("availability_status") != "online")
    out = [_doctor_public_fields(d) for d in doctors]
    if user.role == "admin":
        for d, doc in zip(out, doctors):
            d["verification_status"] = doc.get("verification_status")
    return out


# ---------------------------------------------------------------------------
# Appointment endpoints
# ---------------------------------------------------------------------------
@app.post("/appointments")
def book_appointment(payload: BookAppointmentRequest, user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    if payload.mode not in ("online", "offline"):
        raise HTTPException(400, "mode must be 'online' or 'offline'")
    doctor = db["users"].find_one({"id": payload.doctor_id, "role": "doctor", "verification_status": "verified"})
    if not doctor:
        raise HTTPException(404, "Doctor not found or not yet verified")

    # Validate the requested date/time against the doctor's stated
    # availability window (available_days / available_start_time /
    # available_end_time), so a mother can't request e.g. a Sunday slot from
    # a doctor who only takes weekdays. This is a real constraint check, not
    # just cosmetic -- it runs regardless of appointment mode.
    try:
        appt_date = datetime.date.fromisoformat(payload.appointment_date)
    except ValueError:
        raise HTTPException(400, "appointment_date must be in YYYY-MM-DD format")
    if not TIME_RE.match(payload.appointment_time):
        raise HTTPException(400, "appointment_time must be in HH:MM 24-hour format")
    if appt_date < datetime.date.today():
        raise HTTPException(400, "appointment_date cannot be in the past")

    doctor_days = doctor.get("available_days") or list(ALL_WEEKDAYS)
    weekday_name = ALL_WEEKDAYS[appt_date.weekday()]
    if weekday_name not in doctor_days:
        raise HTTPException(400, f"{doctor['name']} isn't available on {weekday_name}s. Available days: {doctor_days}")

    start = doctor.get("available_start_time") or "09:00"
    end = doctor.get("available_end_time") or "17:00"
    if not (start <= payload.appointment_time <= end):
        raise HTTPException(400, f"{doctor['name']} is available between {start} and {end}")

    appt_id = get_next_id("appointments")
    # Online appointments get a unique video-call room generated up front.
    # This uses Jitsi Meet (meet.jit.si) -- a free, no-signup-required video
    # service -- rather than custom-built WebRTC infrastructure, which is far
    # beyond the scope of this project. Both mother and doctor get the same
    # link and just open it in a browser at the appointment time.
    video_call_url = None
    if payload.mode == "online":
        video_call_url = f"https://meet.jit.si/Amara-Appt{appt_id}-{secrets.token_hex(4)}"

    appt_doc = {
        "id": appt_id,
        "mother_id": user.id, "doctor_id": payload.doctor_id, "mode": payload.mode,
        "appointment_date": payload.appointment_date, "appointment_time": payload.appointment_time,
        "notes": payload.notes, "status": "requested",
        "video_call_url": video_call_url,
        "created_at": datetime.datetime.utcnow(),
    }
    db["appointments"].insert_one(appt_doc)
    return {
        "id": appt_doc["id"], "doctor_name": doctor["name"], "mode": appt_doc["mode"],
        "appointment_date": appt_doc["appointment_date"], "appointment_time": appt_doc["appointment_time"],
        "status": appt_doc["status"], "video_call_url": appt_doc["video_call_url"],
        "qualification": doctor.get("qualification"), "clinic_name": doctor.get("clinic_name"),
        "clinic_address": doctor.get("clinic_address"),
    }


@app.get("/appointments")
def my_appointments(user: DotDict = Depends(require_role("mother")), db=Depends(get_db)):
    appts = list(db["appointments"].find({"mother_id": user.id}).sort("created_at", -1))
    out = []
    for a in appts:
        doctor = db["users"].find_one({"id": a["doctor_id"]})
        out.append({
            "id": a["id"], "doctor_name": doctor["name"] if doctor else "Unknown",
            "specialty": doctor.get("specialty") if doctor else None,
            "qualification": doctor.get("qualification") if doctor else None,
            "clinic_name": doctor.get("clinic_name") if doctor else None,
            "clinic_address": doctor.get("clinic_address") if doctor else None,
            "mode": a["mode"],
            "appointment_date": a.get("appointment_date"), "appointment_time": a.get("appointment_time"),
            "notes": a.get("notes"), "video_call_url": a.get("video_call_url"),
            "status": a["status"],
            "created_at": a["created_at"].isoformat(),
        })
    return out


@app.get("/doctor/appointments")
def doctor_appointments(user: DotDict = Depends(require_role("doctor")), db=Depends(get_db)):
    appts = list(db["appointments"].find({"doctor_id": user.id}).sort("created_at", -1))
    out = []
    for a in appts:
        mother = db["users"].find_one({"id": a["mother_id"]})
        out.append({
            "id": a["id"], "mother_name": mother["name"] if mother else "Unknown",
            "mode": a["mode"],
            "appointment_date": a.get("appointment_date"), "appointment_time": a.get("appointment_time"),
            "notes": a.get("notes"), "video_call_url": a.get("video_call_url"),
            "status": a["status"],
            "created_at": a["created_at"].isoformat(),
        })
    return out


@app.post("/doctor/appointments/{appointment_id}/confirm")
def confirm_appointment(appointment_id: int, user: DotDict = Depends(require_role("doctor")), db=Depends(get_db)):
    appt = db["appointments"].find_one({"id": appointment_id, "doctor_id": user.id})
    if not appt:
        raise HTTPException(404, "Appointment not found")
    db["appointments"].update_one({"id": appointment_id}, {"$set": {"status": "confirmed"}})
    return {"status": "confirmed"}


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------
@app.get("/admin/doctor-verifications")
def list_doctor_verifications(user: DotDict = Depends(require_role("admin")), db=Depends(get_db)):
    doctors = list(db["users"].find({"role": "doctor"}).sort("created_at", -1))
    return [
        {
            "id": d["id"],
            "name": d["name"],
            "email": d["email"],
            "specialty": d.get("specialty"),
            "verification_status": d.get("verification_status") or "unverified",
            "has_document": bool(d.get("verification_doc_base64")),
            "document_original_name": d.get("verification_doc_original_name"),
            "verification_note": d.get("verification_note"),
        }
        for d in doctors
    ]


@app.get("/admin/doctor-verifications/{doctor_id}/document")
def get_verification_document(doctor_id: int, user: DotDict = Depends(require_role("admin")), db=Depends(get_db)):
    doctor = db["users"].find_one({"id": doctor_id, "role": "doctor"})
    if not doctor or not doctor.get("verification_doc_base64"):
        raise HTTPException(404, "No document on file for this doctor")
    content = base64.b64decode(doctor["verification_doc_base64"])
    content_type = doctor.get("verification_doc_content_type") or "application/octet-stream"
    filename = doctor.get("verification_doc_original_name") or "document"
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


class VerificationDecisionRequest(BaseModel):
    note: Optional[str] = None


@app.post("/admin/doctor-verifications/{doctor_id}/approve")
def approve_doctor(doctor_id: int, payload: VerificationDecisionRequest = None,
                    user: DotDict = Depends(require_role("admin")), db=Depends(get_db)):
    doctor = db["users"].find_one({"id": doctor_id, "role": "doctor"})
    if not doctor:
        raise HTTPException(404, "Doctor not found")
    db["users"].update_one(
        {"id": doctor_id},
        {"$set": {"verification_status": "verified", "verification_note": payload.note if payload else None}},
    )
    return {"status": "verified"}


@app.post("/admin/doctor-verifications/{doctor_id}/reject")
def reject_doctor(doctor_id: int, payload: VerificationDecisionRequest,
                   user: DotDict = Depends(require_role("admin")), db=Depends(get_db)):
    doctor = db["users"].find_one({"id": doctor_id, "role": "doctor"})
    if not doctor:
        raise HTTPException(404, "Doctor not found")
    db["users"].update_one(
        {"id": doctor_id},
        {"$set": {"verification_status": "rejected", "verification_note": payload.note}},
    )
    return {"status": "rejected"}


@app.get("/admin/users")
def list_users(user: DotDict = Depends(require_role("admin")), db=Depends(get_db)):
    all_users = list(db["users"].find())
    return [{"id": u["id"], "name": u["name"], "email": u["email"], "role": u["role"]} for u in all_users]


@app.get("/admin/stats")
def stats(user: DotDict = Depends(require_role("admin")), db=Depends(get_db)):
    total_mothers = db["users"].count_documents({"role": "mother"})
    total_doctors = db["users"].count_documents({"role": "doctor"})
    total_messages = db["chat_messages"].count_documents({"sender": "mother"})
    distressed_count = db["chat_messages"].count_documents({"label": 2})
    pending_referrals = db["doctor_referrals"].count_documents({"status": "pending"})
    return {
        "total_mothers": total_mothers,
        "total_doctors": total_doctors,
        "total_messages": total_messages,
        "distressed_messages": distressed_count,
        "pending_referrals": pending_referrals,
    }


# ---------------------------------------------------------------------------
# Frontend (simple demo UI)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("app/templates/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}