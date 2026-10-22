"""
MongoDB data layer for the PPD Support Platform.

Connects via MONGODB_URI (env var). Defaults to a local MongoDB instance.
For a hosted, no-install option, use a free MongoDB Atlas cluster and set:
    MONGODB_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"

Collections mirror the previous SQLAlchemy tables 1:1: users, chat_messages,
community_posts, community_replies, doctor_referrals, appointments -- plus a
`counters` collection used to hand out sequential integer IDs (so the rest of
the app, the Pydantic response models, and the frontend JS -- all of which
assume plain integer IDs -- did not need to change to ObjectId strings).
"""
import os
import datetime
from pymongo import MongoClient, ASCENDING, ReturnDocument

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGODB_DB_NAME", "ppd_app")

client = MongoClient(MONGO_URI)
mongo_db = client[MONGO_DB_NAME]

users = mongo_db["users"]
chat_messages = mongo_db["chat_messages"]
community_posts = mongo_db["community_posts"]
community_replies = mongo_db["community_replies"]
doctor_referrals = mongo_db["doctor_referrals"]
appointments = mongo_db["appointments"]
counters = mongo_db["counters"]


class DotDict(dict):
    """A dict that also supports attribute access (doc.name as well as
    doc["name"]), so the rest of the codebase can keep using the same
    `.attribute` style it used with SQLAlchemy ORM objects, instead of
    rewriting every reference to bracket/dict access. Missing keys resolve
    to None rather than raising, since Mongo documents don't enforce a
    schema the way SQL columns do."""
    def __getattr__(self, item):
        return self.get(item)


def get_next_id(counter_name: str) -> int:
    """Atomic auto-increment, mimicking SQL's integer primary keys."""
    doc = counters.find_one_and_update(
        {"_id": counter_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["seq"]


def assign_doctor_round_robin() -> int | None:
    """Picks the next doctor in rotation for a new referral, so cases are
    spread evenly across doctors instead of piling into one shared queue.
    Only considers verified doctors -- an unverified/pending doctor account
    shouldn't be handed real patient referrals. Returns None if there are no
    verified doctors yet; the caller (main.py's /chat endpoint) stores that
    as doctor_id=None, and list_referrals()/accept_referral() in main.py
    treat doctor_id=None referrals as unclaimed and visible to every doctor,
    so nothing gets silently lost while waiting for the first verification."""
    doctors = list(users.find({"role": "doctor", "verification_status": "verified"}).sort("id", ASCENDING))
    if not doctors:
        return None
    idx_doc = counters.find_one_and_update(
        {"_id": "referral_assignment_index"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    idx = (idx_doc["seq"] - 1) % len(doctors)
    return doctors[idx]["id"]


def init_db():
    """Creates required indexes. Mongo creates collections lazily on first
    insert, so nothing else needs explicit setup."""
    users.create_index([("email", ASCENDING)], unique=True)
    users.create_index([("id", ASCENDING)], unique=True)
    chat_messages.create_index([("mother_id", ASCENDING)])
    community_posts.create_index([("created_at", ASCENDING)])
    community_replies.create_index([("post_id", ASCENDING)])
    doctor_referrals.create_index([("mother_id", ASCENDING)])
    doctor_referrals.create_index([("doctor_id", ASCENDING)])
    appointments.create_index([("mother_id", ASCENDING)])
    appointments.create_index([("doctor_id", ASCENDING)])


def get_db():
    """Kept as a generator for parity with the old SQLAlchemy get_db, so
    `Depends(get_db)` in main.py did not need to change shape."""
    yield mongo_db