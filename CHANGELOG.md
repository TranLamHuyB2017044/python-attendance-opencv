# Changelog

Tất cả các thay đổi đáng kể của dự án sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/vi/1.0.0/),
và dự án tuân theo [Semantic Versioning](https://semver.org/lang/vi/).

## [1.0.0] - 2026-05-28

### ✨ Tính năng mới

- **Hệ thống nhận diện khuôn mặt chuyên nghiệp**
  - Sử dụng InsightFace (ArcFace) cho độ chính xác cao
  - Hỗ trợ nhiều model: buffalo_l, buffalo_s
  - Nhận diện đa góc với Best-of-N voting (gather_frames)
  - Anti-spoofing chống giả mạo khuôn mặt
  - Kiểm tra chất lượng frame (độ mờ, góc xoay)

- **Quản lý Camera IP**
  - Kết nối RTSP ổn định với EZVIZ, Hikvision, Dahua
  - Cấu hình ROI (Region of Interest) để giới hạn vùng nhận diện
  - Hỗ trợ flip ảnh ngang/dọc
  - Cấu hình từ xa qua MongoDB

- **Database hiện đại**
  - MongoDB cho quản lý người dùng và lịch sử chấm công
  - Qdrant Vector DB cho lưu trữ và tìm kiếm vector khuôn mặt
  - Hỗ trợ nhiều company_id riêng biệt

- **UI Quản lý chuyên nghiệp**
  - CustomTkinter giao diện hiện đại
  - Quản lý nhân viên (đăng ký, chỉnh sửa, xóa)
  - Xem lịch sử chấm công theo khoảng thời gian
  - Cấu hình hệ thống trực tiếp từ UI

- **API RESTful**
  - FastAPI backend với các endpoint hoàn chỉnh
  - Tích hợp với HKB Auth Service cho xác thực
  - Webhook để gửi sự kiện chấm công

- **Dịch vụ nền**
  - Service chạy ngầm 24/7
  - Ghi video hàng ngày tự động
  - Upload video lên Google Drive (OAuth2 & Service Account)
  - Ping định kỳ đến Report System
  - Theo dõi thay đổi cài đặt tự động (hot-reload)

- **Thông báo & Logging**
  - Logging với Loguru (file + console)
  - Thông báo qua Telegram Bot
  - Thông báo Desktop (Plyer)
  - Webhook Log Bus cho tích hợp hệ thống khác

- **Tiện ích nâng cao**
  - Quản lý thời gian NTP tự động
  - Single Instance Lock (chạy 1 instance cùng lúc)
  - Hỗ trợ build thành .exe với PyInstaller
  - Test Mode để bỏ qua cooldown

### 🔧 Cấu hình

- File .env hoàn chỉnh với tất cả các tùy chọn
- Cấu hình camera, nhận diện, database, API, webhook
- Cấu hình tự động từ MongoDB (remote config)
- Cấu hình Google Drive (OAuth2 hoặc Service Account)

### 📦 Dependencies

- opencv-python 4.10.0.84
- numpy 1.26.4
- onnxruntime 1.19.2
- insightface 0.7.3
- torch 2.0.1, torchvision 0.15.2
- loguru 0.7.2
- python-dotenv 1.0.1
- scikit-learn 1.5.2
- qdrant-client >= 1.10.1
- pymongo 4.6.1
- fastapi 0.110.0, uvicorn 0.27.1
- customtkinter 5.2.2
- pyinstaller 6.10.0
- bittech-auth-sdk v1.0.2

### 📝 Lưu ý

- Yêu cầu Python 3.10 trở lên
- Khuyến nghị sử dụng venv cho môi trường phát triển
- Đọc README.md để hướng dẫn setup chi tiết
