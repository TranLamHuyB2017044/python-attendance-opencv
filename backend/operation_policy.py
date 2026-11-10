"""Authorization and tenant binding for attendance repository calls."""
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from fastapi import HTTPException

from backend.credentials import device_company_ids

DEVICE_SETTINGS = {
    "camera_ip", "camera_port", "camera_user", "camera_pass", "camera_rtsp_path",
    "camera_rtsp_url", "camera_roi", "detection_cooldown", "anti_spoofing_enabled",
    "enable_telegram_notif", "recognition_threshold", "gather_frames",
}

DEVICE_OPERATIONS = {
    "get_setting", "get_settings", "get_employee", "get_company_name", "get_attendance_status",
    "log_attendance", "update_camera_status", "save_enrollment_image",
    "get_user_company_ids", "get_auth_services_for_companies", "mark_logs_uploaded",
    "find_active_license", "find_latest_license", "save_system_log", "record_system_log",
}

DEVICE_ATTENDANCE_OPERATIONS = {
    "get_employee", "get_company_name", "get_attendance_status", "log_attendance",
    "save_enrollment_image", "get_user_company_ids", "mark_logs_uploaded",
}

INTEGRATION_OPERATIONS = {"get_logs", "get_todays_logs", "get_log_image", "get_all_employees"}

COMPANY_OPERATIONS = {
    "get_setting", "get_settings", "set_setting", "get_camera_status",
    "get_company", "get_company_name", "get_all_companies",
    "get_employee", "get_all_employees", "save_employee", "soft_delete_employee",
    "soft_delete_employees_bulk", "set_employee_active_status", "update_employee_has_face",
    "get_logs", "get_todays_logs", "get_logs_by_user", "get_log_image",
    "save_enrollment_image", "get_enrollment_image",
    "get_auth_service", "get_auth_services_for_user", "save_auth_service", "delete_auth_service",
    "mark_logs_uploaded", "get_unuploaded_logs", "find_active_license", "find_latest_license",
    "save_system_log", "record_system_log", "get_system_logs",
    "get_attendance_status", "log_attendance", "get_user_company_ids", "get_auth_services_for_companies",
}

COMPANY_ARGUMENT = {
    "get_employee", "get_company", "get_company_name", "get_camera_status", "update_camera_status",
    "get_logs", "get_todays_logs", "get_logs_by_user", "get_log_image", "delete_log",
    "log_attendance", "get_attendance_status", "save_employee", "soft_delete_employee",
    "soft_delete_employees_bulk", "set_employee_active_status", "update_employee_has_face",
    "get_enrollment_image", "save_enrollment_image", "delete_enrollment_image",
    "mark_logs_uploaded", "get_unuploaded_logs", "find_active_license", "find_latest_license",
    "save_system_log", "get_system_logs", "record_system_log",
    "get_user_company_ids",
}


def forbidden():
    raise HTTPException(403, "Operation outside the credential scope")


def authorize(repo, actor, operation, arguments):
    kind = actor["kind"]
    if operation in {"verify_login", "update_user_password"}:
        forbidden()
    if kind == "user" and actor["role"] == "admin":
        if operation == "save_auth_service":
            data = arguments["data"]
            if data.get("company_id") is None and actor.get("company_id") not in (None, "", "admin"):
                data["company_id"] = actor["company_id"]
        return None

    company = actor["company_id"]
    allowed = {
        "device": DEVICE_OPERATIONS, "integration": INTEGRATION_OPERATIONS, "user": COMPANY_OPERATIONS,
    }[kind]
    if kind == "user" and actor["role"] == "staff":
        allowed = {"get_camera_status", "get_company_name", "get_company", "get_logs", "get_todays_logs"}
    if operation not in allowed:
        forbidden()

    # Legacy HKB links use UUID as company ID. Additional device connections are
    # granted explicitly by an administrator and never inferred from client input.
    links = {company}
    if kind == "device":
        attendance_companies = set(device_company_ids(actor))
        hkb_companies = set(actor.get("hkb_company_ids", []))
        links = attendance_companies | hkb_companies
    if kind == "user":
        user_id = actor.get("user_id")
        owned = {"company_id": company}
        if user_id is not None:
            owned = {"$or": [owned, {"user_id": user_id, "company_id": {"$exists": False}}]}
        links.update(actor.get("allowed_company_ids") or [])
        integration_links = {company}
        integration_links.update(doc["uuid"] for doc in repo.auth_services.find(owned, {"uuid": 1}) if "uuid" in doc)

    if operation in COMPANY_ARGUMENT:
        target = arguments.get("company_id")
        if target and target != company:
            device_access = kind == "device" and operation in DEVICE_ATTENDANCE_OPERATIONS and target in attendance_companies
            user_access = kind == "user" and operation in {"get_company", "get_company_name", "get_employee", "get_logs", "get_todays_logs"} and target in links
            if not (device_access or user_access):
                forbidden()
        arguments["company_id"] = target or company

    if operation in {"get_setting", "get_settings", "set_setting"}:
        target = arguments.get("username")
        if target not in (None, "GLOBAL", company):
            forbidden()
        arguments["username"] = company
        if operation in {"get_setting", "get_settings"}:
            arguments["excluded_global"] = ("camera_ip", "camera_user", "camera_pass", "camera_rtsp_url", "group_keys")
        if kind == "device":
            keys = arguments["keys"] if operation == "get_settings" else [arguments["key"]]
            if not isinstance(keys, (list, tuple)) or not set(keys) <= DEVICE_SETTINGS:
                forbidden()
            if operation == "get_settings":
                arguments["keys"] = list(keys)
    elif operation == "get_all_employees":
        targets = arguments.get("company_id") or [company]
        if isinstance(targets, str):
            targets = [targets]
        if not targets or not set(targets) <= links:
            forbidden()
        arguments["company_id"] = targets
    elif operation == "get_all_companies":
        return lambda: list(repo.companies.find({"company_id": {"$in": list(links)}}).sort("name", 1))
    elif operation == "get_auth_services_for_user":
        if actor.get("user_id") is not None and arguments["user_id"] != actor["user_id"]:
            forbidden()
        return lambda: list(repo.auth_services.find(owned))
    elif operation == "get_user_company_ids":
        if kind == "device":
            event_company = arguments["company_id"]
            if not repo.get_employee(arguments["user_id"], event_company):
                forbidden()
            # Other attendance companies are independent even if their employee IDs coincide.
            links = {event_company} | hkb_companies
        return lambda: sorted(set(repo.get_user_company_ids(arguments["user_id"])) & links)
    elif operation == "get_auth_services_for_companies":
        requested = set(arguments["company_ids"])
        if not requested <= (integration_links if kind == "user" else links):
            forbidden()
        query = {"uuid": {"$in": list(requested)}}
        if kind == "user":
            return lambda: list(repo.auth_services.find({"$and": [query, owned]}))
        return lambda: list(repo.auth_services.find(query, {
            "_id": 0, "uuid": 1, "endpoint": 1, "key": 1, "user_id": 1, "app_name": 1,
        }))
    elif operation in {"get_auth_service", "delete_auth_service"}:
        identifier = arguments.get("uuid") or arguments.get("system_id")
        if identifier not in integration_links:
            forbidden()
        query = {"$and": [{"uuid": identifier}, owned]}
        if operation == "get_auth_service":
            return lambda: repo.auth_services.find_one(query)

        def delete_owned_connection():
            repo.auth_services.delete_one(query)
        return delete_owned_connection
    elif operation == "save_auth_service":
        data = arguments["data"]
        identifier = data.get("uuid")
        if not isinstance(identifier, str) or not identifier:
            forbidden()
        existing = repo.auth_services.find_one({"uuid": identifier})
        if existing and not repo.auth_services.find_one({"$and": [{"uuid": identifier}, owned]}):
            forbidden()
        data["company_id"] = company
        # Include the ownership condition in the write itself, so a concurrent
        # request cannot replace an integration belonging to a different company.
        return lambda: repo.save_auth_service(data, owner_filter=owned)
    elif operation in {"mark_logs_uploaded", "get_unuploaded_logs"}:
        if arguments["system_id"] not in (integration_links if kind == "user" else links):
            forbidden()
        if kind == "device":
            event_company = arguments["company_id"]
            if arguments["system_id"] not in ({event_company} | hkb_companies):
                forbidden()
            ids = [ObjectId(identifier) for identifier in arguments["log_ids"]]
            if not ids or repo.logs.count_documents({
                "_id": {"$in": ids}, "company_id": event_company, "device_id": actor["_id"],
            }) != len(set(ids)):
                forbidden()
            arguments["device_id"] = actor["_id"]

    if operation in {"find_active_license", "find_latest_license"}:
        arguments["system_id"] = actor["system_id"] if kind == "device" else (arguments.get("system_id") or repo.system_id)
        if operation == "find_active_license":
            arguments["now"] = datetime.now(timezone.utc).replace(tzinfo=None)
        if kind == "device":
            method = getattr(repo, operation)
            def license_status():
                result = method(**arguments)
                return {key: result.get(key) for key in ("status", "expired_at")} if result else None
            return license_status

    if kind == "device":
        if operation in {"get_attendance_status", "log_attendance", "save_enrollment_image"}:
            user_id = arguments["user_id"]
            if not isinstance(user_id, str) or not 1 <= len(user_id) <= 128:
                raise HTTPException(422, "Invalid employee ID")
            if user_id not in ("Unknown", "Spoof"):
                employee = repo.get_employee(user_id, arguments["company_id"])
                if not employee or employee.get("active") is False:
                    if operation == "log_attendance":
                        raise HTTPException(422, "Employee is not active in this company")
                    forbidden()
                if operation == "log_attendance":
                    arguments["user_name"] = employee["name"]
            elif operation == "save_enrollment_image":
                forbidden()
        if operation == "log_attendance":
            captured = arguments.get("captured_at")
            event_id = arguments.get("event_id")
            if (
                not isinstance(captured, datetime) or captured.tzinfo is None
                or captured > datetime.now(timezone.utc) + timedelta(minutes=5)
                or not isinstance(event_id, str) or not 1 <= len(event_id) <= 64
            ):
                raise HTTPException(422, "Invalid camera event timestamp or ID")
            arguments["device_id"] = actor["_id"]
            arguments["status"] = None if arguments["user_id"] not in ("Unknown", "Spoof") else "FAILED"
        elif operation == "update_camera_status":
            if arguments["status"] not in ("running", "waiting_camera") or not isinstance(arguments["camera_connected"], bool):
                raise HTTPException(422, "Invalid camera heartbeat")
            arguments["device_id"] = actor["_id"]
            arguments["last_seen"] = datetime.now(timezone.utc).timestamp()
        elif operation == "record_system_log":
            arguments["device_id"] = actor["_id"]
        elif operation == "save_system_log":
            arguments["system_id"] = actor["system_id"]

    return None
