"""Credentials for management sessions and installed camera services."""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

ISSUER = "attendance-backend"
ACCESS_AUDIENCE = "attendance-management"
ACCESS_AGE = timedelta(minutes=15)
SESSION_AGE = timedelta(days=30)


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def device_company_ids(device):
    return sorted({device["company_id"], *device.get("attendance_company_ids", [])})


def split_secret(value):
    identifier, dot, secret = value.partition(".")
    if not dot or not identifier or not secret or len(value) > 512:
        raise HTTPException(401, "Invalid credential")
    return identifier, digest(secret)


class Credentials:
    def __init__(self, db, repository):
        self.repo = repository
        self.users = db["users"]
        self.sessions = db["attendance_sessions"]
        self.devices = db["attendance_devices"]
        self.limits = db["attendance_login_limits"]
        self.resets = db["attendance_password_resets"]
        self.signing_key = os.getenv("ATTENDANCE_JWT_SECRET", "")
        if len(self.signing_key.encode("utf-8")) < 32:
            raise RuntimeError("Set ATTENDANCE_JWT_SECRET to a random server secret of at least 32 bytes")
        self.integration_token = os.getenv("ATTENDANCE_INTEGRATION_TOKEN", "")
        self.integration_company = os.getenv("ATTENDANCE_INTEGRATION_COMPANY", "")
        self.public_url = os.getenv("ATTENDANCE_PUBLIC_URL", "").rstrip("/")

    def ensure_indexes(self):
        self.sessions.create_index("expires_at", expireAfterSeconds=0)
        self.limits.create_index("expires_at", expireAfterSeconds=0)
        self.resets.create_index("expires_at", expireAfterSeconds=0)
        self.devices.create_index("company_id")

    def sign(self, audience, expiry, **claims):
        return jwt.encode({
            "iss": ISSUER, "aud": audience, "iat": datetime.now(timezone.utc),
            "exp": expiry.replace(tzinfo=timezone.utc), **claims,
        }, self.signing_key, algorithm="HS256")

    def decode(self, value, audience):
        try:
            return jwt.decode(
                value, self.signing_key, algorithms=["HS256"], issuer=ISSUER, audience=audience,
                options={"require": ["iss", "aud", "iat", "exp", "sub"]},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(401, "Invalid or expired token") from None

    def throttle(self, key, limit):
        current = now()
        bucket = current.replace(minute=current.minute - current.minute % 5, second=0, microsecond=0)
        query = {"_id": f"{bucket.isoformat()}:{digest(key)}"}
        try:
            count = self.limits.find_one_and_update(
                query, {"$inc": {"count": 1}, "$setOnInsert": {"expires_at": bucket + timedelta(minutes=5)}},
                upsert=True, return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            count = self.limits.find_one_and_update(
                query, {"$inc": {"count": 1}}, return_document=ReturnDocument.AFTER,
            )
        if count["count"] > limit:
            raise HTTPException(429, "Too many authentication attempts", headers={"Retry-After": "300"})

    def user(self, username):
        user = self.users.find_one({"username": username}, {"password": 0})
        if not user or user.get("active") is False:
            raise HTTPException(401, "Invalid or inactive account")
        user["role"] = str(user.get("role", "")).lower()
        if user["role"] not in {"admin", "company", "staff"}:
            raise HTTPException(401, "Invalid account role")
        if user["role"] != "admin" and not user.get("company_id"):
            raise HTTPException(401, "Account has no company")
        return user

    def tokens(self, session_id, user, secret):
        return {
            "access_token": self.sign(ACCESS_AUDIENCE, now() + ACCESS_AGE, sub=user["username"], sid=session_id),
            "refresh_token": f"{session_id}.{secret}",
            "user": {key: user.get(key) for key in ("username", "company_id", "role", "user_id")},
        }

    def login(self, username, password, host):
        username = username.strip()
        self.throttle("ip:" + host, 60)
        self.throttle("account:" + username.casefold(), 12)
        if not self.repo.verify_login(username, password):
            raise HTTPException(401, "Invalid credentials")
        user = self.user(username)
        session_id, secret = uuid4().hex, secrets.token_urlsafe(32)
        self.sessions.insert_one({
            "_id": session_id, "username": username, "refresh_hash": digest(secret),
            "auth_version": user.get("auth_version", 0), "account_id": user["_id"],
            "expires_at": now() + SESSION_AGE,
        })
        return self.tokens(session_id, user, secret)

    def refresh(self, value):
        session_id, supplied_hash = split_secret(value)
        secret = secrets.token_urlsafe(32)
        valid = {"_id": session_id, "expires_at": {"$gt": now()}, "revoked_at": None}
        session = self.sessions.find_one_and_update(
            {**valid, "refresh_hash": supplied_hash},
            {"$set": {"refresh_hash": digest(secret)}, "$push": {"used_hashes": supplied_hash}},
        )
        if not session:
            self.sessions.update_one({**valid, "used_hashes": supplied_hash}, {"$set": {"revoked_at": now()}})
            raise HTTPException(401, "Invalid or expired refresh token")
        user = self.user(session["username"])
        if session.get("account_id") != user["_id"] or session["auth_version"] != user.get("auth_version", 0):
            raise HTTPException(401, "Session revoked")
        return self.tokens(session_id, user, secret)

    def authenticate(self, authorization):
        scheme, separator, value = authorization.partition(" ")
        if not separator or not value:
            raise HTTPException(401, "Authentication required")
        if scheme.lower() == "device":
            identifier, key_hash = split_secret(value)
            device = self.devices.find_one({"_id": identifier, "revoked_at": None})
            if device:
                current = hmac.compare_digest(key_hash, device["key_hash"])
                previous = (
                    device.get("previous_until") is not None and device["previous_until"] > now()
                    and hmac.compare_digest(key_hash, device["previous_hash"])
                )
                if current or previous:
                    return {"kind": "device", **device}
            raise HTTPException(401, "Invalid device key")
        if scheme.lower() == "integration" and self.integration_token and self.integration_company:
            if hmac.compare_digest(value.encode(), self.integration_token.encode()):
                return {"kind": "integration", "company_id": self.integration_company}
        if scheme.lower() != "bearer":
            raise HTTPException(401, "Invalid credential type")
        claims = self.decode(value, ACCESS_AUDIENCE)
        session = self.sessions.find_one({
            "_id": claims.get("sid"), "username": claims["sub"], "expires_at": {"$gt": now()}, "revoked_at": None,
        })
        user = self.user(claims["sub"])
        if not session or session.get("account_id") != user["_id"] or session["auth_version"] != user.get("auth_version", 0):
            raise HTTPException(401, "Session revoked")
        return {"kind": "user", **user, "session_id": session["_id"]}

    def create_device(self, company_id, name, system_id, hkb_company_ids, username, attendance_company_ids):
        identifier, secret = uuid4().hex, secrets.token_urlsafe(32)
        self.devices.insert_one({
            "_id": identifier, "company_id": company_id, "name": name, "key_hash": digest(secret),
            "system_id": system_id or self.repo.system_id, "hkb_company_ids": sorted(set(hkb_company_ids)),
            "attendance_company_ids": attendance_company_ids,
            "created_at": now(), "created_by": username,
        })
        return {"device_id": identifier, "api_key": f"{identifier}.{secret}"}

    def rotate_device(self, identifier):
        secret = secrets.token_urlsafe(32)
        result = self.devices.update_one({"_id": identifier, "revoked_at": None}, [
            {"$set": {
                "previous_hash": "$key_hash", "previous_until": now() + timedelta(minutes=15),
                "key_hash": digest(secret), "rotated_at": now(),
            }},
        ])
        if not result.matched_count:
            raise HTTPException(404, "Active device not found")
        return {"device_id": identifier, "api_key": f"{identifier}.{secret}"}

    def reset_code(self, username):
        self.user(username)
        secret = secrets.token_urlsafe(32)
        self.resets.replace_one(
            {"_id": username}, {"_id": username, "code_hash": digest(secret), "expires_at": now() + timedelta(minutes=30)},
            upsert=True,
        )
        return secret

    def reset_password(self, username, code, password, host):
        self.throttle("reset:" + host, 20)
        match = {"_id": username, "code_hash": digest(code), "expires_at": {"$gt": now()}}
        record = self.resets.find_one_and_delete(match)
        if not record:
            raise HTTPException(401, "Invalid or expired reset code")
        if not self.repo.update_user_password(username, password):
            raise HTTPException(503, "Password update failed. Request a new reset code")
