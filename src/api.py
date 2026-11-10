from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from src.attendance.mongodb_mgr import AttendanceBackendClient
from src.config import ApiConfig, AttendanceBackendConfig
import uvicorn
import json
import hmac
import re
from loguru import logger

app = FastAPI(title="Face Attendance External API")

mongo_db = AttendanceBackendClient()
mongo_db.integration_token = AttendanceBackendConfig.INTEGRATION_TOKEN

def integration_company(authorization):
    expected = AttendanceBackendConfig.EXTERNAL_API_TOKEN
    company = AttendanceBackendConfig.INTEGRATION_COMPANY
    if not expected or not company or not mongo_db.integration_token:
        raise HTTPException(503, "Attendance integration is disabled")
    if not hmac.compare_digest(authorization.encode(), f"Bearer {expected}".encode()):
        raise HTTPException(401, "Invalid integration token")
    return company


class LogQueryRequest(BaseModel):
    company_id: str
    date: Optional[str] = None # Format: YYYY-MM-DD

@app.post("/api/attendance/logs")
def get_attendance_logs(
    request: LogQueryRequest,
    authorization: str = Header(..., description="Bearer <token>")
):
    """
    Fetch attendance logs via POST body with Token Authentication.
    """
    if integration_company(authorization) != request.company_id:
        raise HTTPException(403, "Company is outside the integration scope")

    target_date = request.date
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Query logs through the attendance backend
        logs = mongo_db.get_logs(company_id=request.company_id, date=target_date)
        
        results = []
        for log in logs:
            results.append({
                "id": str(log["_id"]),
                "user_id": log["user_id"],
                "user_name": log["user_name"],
                "timestamp": log["timestamp"],
                "date": log["date"],
                "status": log["status"],
                "image_url": f"{ApiConfig.BASE_URL}/api/attendance/image/{str(log['_id'])}"
            })
            
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/image/{log_id}")
def get_log_image(log_id: str, authorization: str = Header(default="")):
    """
    Serve the attendance image (WebP) from MongoDB.
    """
    integration_company(authorization)
    try:
        if not re.fullmatch(r"[0-9a-fA-F]{24}", log_id):
            raise HTTPException(status_code=400, detail="Invalid Log ID format")
            
        img_bytes = mongo_db.get_log_image(log_id)
        if not img_bytes:
            raise HTTPException(status_code=404, detail="Image not found")
            
        return Response(content=img_bytes, media_type="image/webp")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/share/employees")
def get_shared_employees(
    authorization: str = Header(..., description="Bearer <token>")
):
    """
    Fetch all employees for sharing with other systems via HKB Auth flow.
    """
    cid = integration_company(authorization)

    try:
        # Fetch employees through the attendance backend
        employees = mongo_db.get_all_employees(company_id=[cid])
        results = []
        for emp in employees:
            results.append({
                "user_id": emp["user_id"],
                "name": emp["name"],
                "group_id": emp.get("group_id", cid),
                "birthday": emp.get("birthday")
            })
        
        logger.info(f"API: Sharing {len(results)} employees for company {cid}")
        logger.debug(f"Employee Data JSON: {json.dumps(results, ensure_ascii=False)}")

        return {
            "success": True,
            "status": "success",
            "message": f"Found {len(results)} employees for company {cid}",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/api/hr/users", methods=["GET", "POST"])
def get_hr_users_api(
    authorization: str = Header(..., description="Bearer <token>")
):
    """
    Fetch all users for HR purposes, specifically for 'tester' environment.
    """
    cid = integration_company(authorization)

    try:
        # Fetch employees through the attendance backend
        employees = mongo_db.get_all_employees(company_id=[cid])
        results = []
        for emp in employees:
            results.append({
                "user_id": emp["user_id"],
                "name": emp["name"],
                "company_id": emp["company_id"],
                "birthday": emp.get("birthday")
            })
        
        logger.info(f"API: Retrieved {len(results)} users for HR/Tester (Company: {cid})")
        logger.debug(f"HR User Data JSON: {json.dumps(results, ensure_ascii=False)}")

        return {
            "success": True,
            "status": "success",
            "message": f"Retrieved {len(results)} users for HR/Tester",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_api():
    """Start the FastAPI server."""
    uvicorn.run(app, host=ApiConfig.HOST, port=ApiConfig.PORT)

if __name__ == "__main__":
    start_api()
