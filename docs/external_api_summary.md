# External API Summary - BitTech System Report

Tài liệu này tóm tắt các API mà các hệ thống bên ngoài (ví dụ: Python Attendance App) có thể gọi để gửi log và duy trì trạng thái hoạt động trên hệ thống giám sát.

---

## 1. Gửi Log (Report Log)

Dùng để gửi thông báo, lỗi hoặc thông tin vận hành từ các hệ thống khác lên Dashboard.

- **Endpoint**: `/api/v1/report-log`
- **Method**: `POST`
- **Authentication**: Dựa trên `Authorization: Bearer <token>` và `system_id` trong body. Hệ thống tự động đối soát `api_key` nội bộ.
- **Content-Type**: `application/json`

### Headers bắt buộc

| Header | Bắt buộc | Mô tả |
| :--- | :--- | :--- |
| `Authorization` | **Có** | Bearer Token được cấp sau khi login. |
| `X-Trace-Id` | **Có** | Chuỗi định danh (UUID hoặc mã bất kỳ do client cấp) dùng để theo dõi (trace) request tương ứng với lỗi này. |

### Tham số (Body)

| Tham số | Kiểu dữ liệu | Bắt buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `system_id` | String (UUID) | **Có** | ID duy nhất của hệ thống (UUID được cấp khi hệ thống được tạo/sync). |
| `message` | String | **Có** | Nội dung thông báo hoặc lỗi chi tiết. |
| `status_code` | Number/String | **Có** | Mã lỗi HTTP hoặc mã trạng thái tùy chỉnh của ứng dụng. |
| `type` | String | Không | Loại log (`ERROR` hoặc `INFO`). Mặc định: `INFO`. |
| `priority` | String | Không | Mức độ ưu tiên (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`). Mặc định: `MEDIUM`. |
| `devices_info` | Object | Không | Thông tin chi tiết về thiết bị (ví dụ: CPU, RAM, Disk, OS, IP, v.v.). |

### Ví dụ Request (cURL)
```bash
curl -X POST http://your-domain.com/api/v1/report-log \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN_HERE" \
-H "X-Trace-Id: trace-12345-abcde" \
-d '{
  "system_id": "848bbb0f-6584-4d74-8fe1-e44f1de83885",
  "message": "Connection loss detected with RTSP camera",
  "status_code": 503
}'
```

### Phản hồi (Response)
- **201 Created**: `{ "success": true, "message": "Log created successfully" }`
- **400 Bad Request**: Trả về khi thiếu tham số bắt buộc.
- **404 Not Found**: 
```json
{
  "success": false,
  "message": "Hệ thống với ID '848bbb0f-6584-4d74-8fe1-e44f1de83885' không tồn tại trên Dashboard. Vui lòng kiểm tra lại cấu hình."
}
```

---

## 2. Heartbeat (System Ping)

Dùng để cập nhật thời gian hoạt động cuối cùng của hệ thống. Nếu hệ thống ngừng ping quá thời gian quy định (mặc định 5 phút), Dashboard sẽ hiển thị trạng thái **OFFLINE** và gửi cảnh báo email.

- **Endpoint**: `/api/v1/systems/ping`
- **Method**: `POST`
- **Authentication**: Không (Dựa trên `system_id` hợp lệ).
- **Content-Type**: `application/json`

### Tham số (Body)

| Tham số | Kiểu dữ liệu | Bắt buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `system_id` | String (UUID) | **Có** | ID duy nhất của hệ thống. |
| `ping_time` | ISO String | Không | Thời gian ghi nhận ping (ISO 8601). Nếu để trống sẽ dùng thời gian hiện tại của server. |
| `devices_info` | Object | Không | Cập nhật thông tin phần cứng hoặc trạng thái hiện tại của thiết bị. |
| `system_name` | String | Không | Không bắt buộc, Dashboard ưu tiên dùng tên từ DB. |

### Ví dụ Request (cURL)
```bash
curl -X POST http://your-domain.com/api/v1/systems/ping \
-H "Content-Type: application/json" \
-d '{
  "system_id": "848bbb0f-6584-4d74-8fe1-e44f1de83885",
  "devices_info": {
    "status": "Online",
    "uptime": "2 days"
  }
}'
```

### Phản hồi (Response)
- **200 OK**: 
```json
{
  "success": true,
  "message": "Pong!",
  "data": {
    "system_name": "FACE AI CHECKING",
    "last_check_time": "2024-04-09T02:00:00.000Z",
    "connection_status": "CONNECTED"
  }
}
```
- **404 Not Found**: 
```json
{
  "success": false,
  "message": "Hệ thống với ID '848bbb0f-6584-4d74-8fe1-e44f1de83885' không tồn tại trên Dashboard. Vui lòng kiểm tra lại cấu hình."
}
```

---

## Ghi chú Quan trọng

1. **System ID**: Đảm bảo `system_id` gửi lên trùng khớp với UUID trong database của hệ thống giám sát. Nếu không, các yêu cầu sẽ bị từ chối với lỗi 404.
2. **Auto-Recovery**: Nếu một hệ thống đang ở trạng thái `DISCONNECTED` và thực hiện gọi API `ping` thành công, hệ thống sẽ tự động chuyển sang trạng thái `CONNECTED` và gửi thông báo khôi phục (Recovery) qua email.
3. **Devices Info**: Dữ liệu trong `devices_info` là tùy biến (Object), có thể chứa bất kỳ thông tin nào hữu ích cho việc giám sát từ xa.
