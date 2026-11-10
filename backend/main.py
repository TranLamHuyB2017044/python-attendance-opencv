import inspect
import json
import logging
import os
import sys
from asyncio import to_thread
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from threading import Lock

import uvicorn
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.attendance_repository import AttendanceRepository
from backend.credentials import Credentials, device_company_ids, now
from backend.operation_policy import authorize, forbidden
from src.attendance_transport import OPERATIONS, decode_argument, encode_value

logger = logging.getLogger("uvicorn.error.attendance")


@asynccontextmanager
async def lifespan(app: FastAPI):
    uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DB")
    if not uri or not database_name:
        raise RuntimeError("Set MONGODB_URI and MONGODB_DB before starting the backend")

    # Share the MongoDB connection pool across requests in this process.
    try:
        client = MongoClient(
            uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, timeoutMS=5000
        )
    except (PyMongoError, ValueError) as error:
        raise RuntimeError(f"MongoDB client initialization failed ({type(error).__name__})") from None

    with client:
        try:
            app.state.mongo_db = client[database_name]
        except (PyMongoError, ValueError):
            raise RuntimeError("Invalid MONGODB_DB") from None
        app.state.attendance = AttendanceRepository(app.state.mongo_db)
        app.state.credentials = Credentials(app.state.mongo_db, app.state.attendance)
        app.state.indexes_ready = False
        app.state.indexes_lock = Lock()

        try:
            await to_thread(app.state.mongo_db.command, "ping")
        except PyMongoError as error:
            logger.warning("MongoDB startup ping failed (%s). Check /health/ready.", type(error).__name__)
        else:
            logger.info("MongoDB startup ping succeeded (database: %s).", database_name)

        yield


app = FastAPI(
    title="Attendance Backend",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(request: Request):
    try:
        request.app.state.mongo_db.command("ping")
        ensure_indexes(request)
    except PyMongoError as error:
        logger.warning("MongoDB readiness check failed (%s)", type(error).__name__)
        raise HTTPException(503, "MongoDB unavailable") from None
    return {"status": "ok"}


def ensure_indexes(request: Request):
    state = request.app.state
    if not state.indexes_ready:
        with state.indexes_lock:
            if not state.indexes_ready:
                state.attendance.ensure_indexes()
                state.credentials.ensure_indexes()
                state.indexes_ready = True


async def read_operation(request: Request):
    try:
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > 8 * 1024 * 1024:
                raise HTTPException(413, "Attendance request is too large")
            payload.extend(chunk)
        body = json.loads(payload, object_hook=decode_argument)
        if not isinstance(body["args"], list) or not isinstance(body["kwargs"], dict):
            raise ValueError
        return body["args"], body["kwargs"]
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, "Invalid attendance operation arguments") from None


def encode_result(value):
    if isinstance(value, ObjectId):
        return str(value)
    return encode_value(value)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    return JSONResponse({"detail": "Invalid request fields"}, status_code=422)


@app.exception_handler(PyMongoError)
async def mongo_error(request: Request, error: PyMongoError):
    logger.warning("MongoDB operation failed (%s)", type(error).__name__)
    return JSONResponse({"detail": "MongoDB unavailable"}, status_code=503)


def credentials(request: Request):
    ensure_indexes(request)
    return request.app.state.credentials


def authenticated(authorization: str = Header(default=""), store=Depends(credentials)):
    return store.authenticate(authorization)


def administrator(actor=Depends(authenticated)):
    if actor["kind"] != "user" or actor["role"] != "admin":
        forbidden()
    return actor


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class Refresh(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=512)


class PasswordChange(Login):
    new_password: str = Field(min_length=8, max_length=72)


class ResetRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)


class PasswordReset(ResetRequest):
    code: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=72)


class DeviceCompanyScope(BaseModel):
    attendance_company_ids: list[str] = Field(default_factory=list, max_length=100)


class DeviceRegistration(DeviceCompanyScope):
    company_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    system_id: str | None = Field(default=None, max_length=128)
    hkb_company_ids: list[str] = Field(default_factory=list)


def peer(request):
    return request.client.host if request.client else "unknown"


def check_password_length(password):
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(422, "Password must fit in 72 UTF-8 bytes")


@app.post("/auth/login")
def login(body: Login, request: Request, store=Depends(credentials)):
    return store.login(body.username, body.password, peer(request))


@app.post("/auth/refresh")
def refresh(body: Refresh, store=Depends(credentials)):
    return store.refresh(body.refresh_token)


@app.get("/auth/me")
def current_user(actor=Depends(authenticated)):
    if actor["kind"] != "user":
        forbidden()
    return {key: actor.get(key) for key in ("username", "role", "company_id")}


@app.post("/auth/logout")
def logout(store=Depends(credentials), actor=Depends(authenticated)):
    if actor["kind"] != "user":
        forbidden()
    store.sessions.update_one({"_id": actor["session_id"]}, {"$set": {"revoked_at": now()}})
    return {"ok": True}


@app.post("/auth/change-password")
def change_password(body: PasswordChange, request: Request, store=Depends(credentials)):
    store.throttle("change-ip:" + peer(request), 30)
    store.throttle("change-account:" + body.username.casefold(), 12)
    check_password_length(body.new_password)
    if not store.repo.verify_login(body.username, body.password):
        raise HTTPException(400, "Current password is incorrect")
    store.user(body.username)
    if not store.repo.update_user_password(body.username, body.new_password):
        raise HTTPException(503, "Password update failed")
    return {"ok": True}


@app.post("/management/password-reset")
def create_reset(body: ResetRequest, store=Depends(credentials), actor=Depends(administrator)):
    return {"reset_code": store.reset_code(body.username)}


@app.post("/auth/reset-password")
def reset_password(body: PasswordReset, request: Request, store=Depends(credentials)):
    check_password_length(body.new_password)
    store.reset_password(body.username, body.code, body.new_password, peer(request))
    return {"ok": True}


@app.get("/management/devices")
def list_devices(store=Depends(credentials), actor=Depends(administrator)):
    projection = {"key_hash": 0, "previous_hash": 0}
    result = []
    for device in store.devices.find({}, projection).sort("name", 1):
        device["device_id"] = device.pop("_id")
        device["attendance_company_ids"] = device_company_ids(device)
        heartbeat = store.repo.system_status.find_one({
            "company_id": device["company_id"], "device_id": device["device_id"], "type": "camera_service",
        })
        if heartbeat:
            device["last_seen_at"] = datetime.fromtimestamp(heartbeat["last_seen"], tz=timezone.utc)
        result.append(device)
    return result


@app.post("/management/devices", status_code=201)
def register_device(body: DeviceRegistration, store=Depends(credentials), actor=Depends(administrator)):
    companies = validate_device_companies(store, body.company_id, body.attendance_company_ids)
    if len(body.hkb_company_ids) > 100 or any(not item or len(item) > 128 for item in body.hkb_company_ids):
        raise HTTPException(422, "Invalid HKB connection list")
    return store.create_device(body.company_id, body.name, body.system_id, body.hkb_company_ids, actor["username"], companies)


def validate_device_companies(store, primary, additional):
    companies = sorted({primary, *additional})
    if any(not item or len(item) > 128 for item in companies):
        raise HTTPException(422, "Invalid attendance company list")
    existing = store.repo.companies.distinct("company_id", {"company_id": {"$in": companies}})
    if set(existing) != set(companies):
        raise HTTPException(404, "Company not found")
    return companies


@app.patch("/management/devices/{device_id}/companies")
def update_device_companies(device_id: str, body: DeviceCompanyScope, store=Depends(credentials), actor=Depends(administrator)):
    query = {"_id": device_id, "revoked_at": None}
    device = store.devices.find_one(query, {"company_id": 1})
    if not device:
        raise HTTPException(404, "Active device not found")
    companies = validate_device_companies(store, device["company_id"], body.attendance_company_ids)
    if not store.devices.update_one(query, {"$set": {"attendance_company_ids": companies}}).matched_count:
        raise HTTPException(404, "Active device not found")
    return {"attendance_company_ids": companies}


@app.post("/management/devices/{device_id}/rotate")
def rotate_device(device_id: str, store=Depends(credentials), actor=Depends(administrator)):
    return store.rotate_device(device_id)


@app.post("/management/devices/{device_id}/revoke")
def revoke_device(device_id: str, store=Depends(credentials), actor=Depends(administrator)):
    result = store.devices.update_one({"_id": device_id}, {"$set": {"revoked_at": now()}})
    if not result.matched_count:
        raise HTTPException(404, "Device not found")
    return {"ok": True}


@app.get("/devices/self")
def device_identity(actor=Depends(authenticated)):
    if actor["kind"] != "device":
        forbidden()
    return {
        "device_id": actor["_id"], "company_id": actor["company_id"],
        "attendance_company_ids": device_company_ids(actor),
        "name": actor["name"], "system_id": actor["system_id"],
    }


@app.get("/media/attendance/{log_id}")
def attendance_image(log_id: str, token: str, store=Depends(credentials)):
    claims = store.decode(token, "attendance-image")
    if claims["sub"] != log_id or not claims.get("company_id"):
        forbidden()
    image = store.repo.get_log_image(log_id, company_id=claims["company_id"])
    if not image:
        raise HTTPException(404, "Image not found")
    return Response(image, media_type="image/webp", headers={"Cache-Control": "private, no-store"})


@app.post("/internal/attendance/{operation}")
def run_operation(operation: str, request: Request, actor=Depends(authenticated), arguments=Depends(read_operation)):
    if operation not in OPERATIONS:
        raise HTTPException(404, "Unknown attendance operation")
    repo = request.app.state.attendance
    method = getattr(repo, operation)
    args, kwargs = arguments
    try:
        bound = inspect.signature(method).bind(*args, **kwargs)
        values = bound.arguments
        if {"device_id", "owner_filter", "excluded_global"} & values.keys():
            forbidden()
        if operation == "log_attendance" and actor["kind"] != "device" and {"event_id", "captured_at"} & values.keys():
            forbidden()
        handler = authorize(repo, actor, operation, values)
        result = handler() if handler else method(**values)
        if operation == "log_attendance":
            status, event = result
            if status is None:
                raise HTTPException(503, "Attendance could not be saved")
            if event and event.get("company_id"):
                store = request.app.state.credentials
                log_id = str(event["_id"])
                token = store.sign("attendance-image", now() + timedelta(minutes=30), sub=log_id, company_id=event["company_id"])
                base_url = store.public_url or str(request.base_url).rstrip("/")
                event["image_url"] = f"{base_url}/media/attendance/{log_id}?{urlencode({'token': token})}"
        response = {"result": result}
        if actor["kind"] == "device":
            response["attendance_company_ids"] = device_company_ids(actor)
        return Response(json.dumps(response, default=encode_result), media_type="application/json")
    except (TypeError, ValueError, InvalidId):
        raise HTTPException(422, "Invalid attendance operation values") from None


if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parent
    env_file = backend_dir / ".env"
    if not env_file.is_file():
        env_file = backend_dir.parent / ".env"
    from dotenv import load_dotenv

    load_dotenv(env_file)
    uvicorn.run(
        app, host=os.getenv("ATTENDANCE_BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("ATTENDANCE_BACKEND_PORT", "8001")), access_log=False,
    )
