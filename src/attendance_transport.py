"""Shared wire format for the private attendance API."""
from base64 import b64decode, b64encode
from datetime import datetime

OPERATIONS = frozenset("""
    get_setting get_settings set_setting seed_setting get_attendance_status log_attendance
    get_logs get_todays_logs get_log_image get_logs_by_user delete_log
    verify_login create_user get_all_cloud_users get_user_by_username get_user_by_email
    update_user_password delete_user
    get_employee get_all_employees save_employee get_user_company_ids delete_employee
    delete_employees_bulk soft_delete_employee soft_delete_employees_bulk
    update_employee_has_face set_employee_active_status
    get_company get_all_companies get_company_name create_company delete_company
    save_enrollment_image get_enrollment_image delete_enrollment_image
    delete_enrollment_images_by_user
    get_auth_service get_auth_services_for_user get_auth_services_for_companies
    save_auth_service delete_auth_service
    find_latest_license find_active_license save_pending_license mark_license_verified
    update_license_status get_unuploaded_logs mark_logs_uploaded
    get_camera_status update_camera_status save_system_log get_system_logs record_system_log
""".split())


def encode_value(value):
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes": b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported attendance value: {type(value).__name__}")


def decode_value(value):
    if value.keys() == {"$datetime"}:
        return datetime.fromisoformat(value["$datetime"])
    if value.keys() == {"$bytes"}:
        return b64decode(value["$bytes"], validate=True)
    return value


def decode_argument(value):
    decoded = decode_value(value)
    if isinstance(decoded, dict) and any(key.startswith("$") for key in decoded):
        raise ValueError("Database operators are not accepted by the attendance API")
    return decoded
