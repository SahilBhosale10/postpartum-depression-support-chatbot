"""
Run once after first starting the app to create an admin account and a couple
of demo doctors:  python3 seed.py
"""
import datetime
from app.database import mongo_db, init_db, get_next_id
from app.auth import hash_password

init_db()

seed_users = [
    {"name": "Admin", "email": "admin@ppd.com", "password": "admin123", "role": "admin", "specialty": None},
    {"name": "Dr. Anika Rao", "email": "anika@ppd.com", "password": "doctor123", "role": "doctor", "specialty": "Perinatal Psychiatry"},
    {"name": "Dr. Marcus Lee", "email": "marcus@ppd.com", "password": "doctor123", "role": "doctor", "specialty": "Clinical Psychology"},
]

for u in seed_users:
    existing = mongo_db["users"].find_one({"email": u["email"]})
    if existing:
        print(f"Skipping (already exists): {u['email']}")
        continue
    mongo_db["users"].insert_one({
        "id": get_next_id("users"),
        "name": u["name"], "email": u["email"],
        "hashed_password": hash_password(u["password"]),
        "role": u["role"], "specialty": u["specialty"],
        # Seeded demo doctors are pre-verified so the demo works out of the
        # box -- real doctors signing up through /auth/register start
        # "unverified" and need to upload a document + get admin approval
        # (see main.py's /doctor/verification/document and
        # /admin/doctor-verifications/{id}/approve).
        "verification_status": "verified" if u["role"] == "doctor" else None,
        "verification_doc_filename": None,
        "verification_doc_original_name": None,
        "verification_note": "Seeded demo account -- pre-verified." if u["role"] == "doctor" else None,
        # Seeded demo doctors start online so the availability feature is
        # visible immediately in a demo; real doctors default to offline.
        "availability_status": "online" if u["role"] == "doctor" else None,
        "created_at": datetime.datetime.utcnow(),
    })
    print(f"Created {u['role']}: {u['email']} / {u['password']}")

print("\nSeeding complete.")