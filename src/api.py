from fastapi import FastAPI, HTTPException, Header, Response, Body
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from src.attendance.mongodb_mgr import mongo_db
from src.config import ApiConfig, CAPTURES_DIR
import uvicorn
from bson.objectid import ObjectId

app = FastAPI(title="Face Attendance External API")

# Serve local captures as static files
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

# Hardcoded tokens for demonstration
# In production, these should be generated and stored in MongoDB
VALID_TOKENS = {
    "749b2b7e-40d2-4946-b346-8d76b4242688": {"company_id": "default_company", "role": "admin"},
    "company-a-token": {"company_id": "company_a", "role": "user"}
}

class LogQueryRequest(BaseModel):
    company_id: str
    date: Optional[str] = None # Format: YYYY-MM-DD

@app.post("/api/attendance/logs")
async def get_attendance_logs(
    request: LogQueryRequest,
    authorization: str = Header(..., description="Bearer <token>")
):
    """
    Fetch attendance logs via POST body with Token Authentication.
    """
    # 1. Simple Token Validation
    token = authorization.replace("Bearer ", "")
    user_context = VALID_TOKENS.get(token)
    
    if not user_context:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
        
    # 2. Authorization: Check if user has access to this company
    # Admin can see all, but here we restriction to the token's company for security
    if user_context["role"] != "admin" and user_context["company_id"] != request.company_id:
        raise HTTPException(status_code=403, detail="You do not have access to this company's logs")

    target_date = request.date
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Query logs from MongoDB
        query = {"company_id": request.company_id, "date": target_date}
        logs_cursor = mongo_db.logs.find(query).sort("_id", -1)
        
        results = []
        for log in logs_cursor:
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
async def get_log_image(log_id: str):
    """
    Serve the attendance image (WebP) from MongoDB.
    """
    try:
        if not ObjectId.is_valid(log_id):
            raise HTTPException(status_code=400, detail="Invalid Log ID format")
            
        img_bytes = mongo_db.get_log_image(log_id)
        if not img_bytes:
            raise HTTPException(status_code=404, detail="Image not found")
            
        return Response(content=img_bytes, media_type="image/webp")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_api():
    """Start the FastAPI server."""
    uvicorn.run(app, host=ApiConfig.HOST, port=ApiConfig.PORT)

if __name__ == "__main__":
    start_api()
