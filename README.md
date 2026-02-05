# Python Face Attendance System (InsightFace + OpenCV)

Hệ thống chấm công bằng khuôn mặt chuyên nghiệp sử dụng InsightFace cho độ chính xác cao và OpenCV để xử lý video từ Camera IP (EZVIZ, Hikvision, Dahua).

## 🚀 Kiến trúc dự án
```text
python-attendance-opencv/
├── src/
│   ├── camera/        # Kết nối RTSP, xử lý frame ổn định
│   ├── recognition/   # InsightFace (ArcFace) cho Detection & Embedding
│   ├── attendance/    # Quản lý Database người dùng và so khớp Cosine
│   ├── utils/         # Logging (Loguru) và tiện ích khác
│   ├── config.py      # Cấu hình hệ thống qua .env
│   └── main.py        # Entry point của ứng dụng
├── data/              # Chứa ảnh ví dụ (nếu có)
├── embeddings/        # Lưu trữ vector khuôn mặt (users.pkl)
├── logs/              # Log file (app.log)
├── models/            # Nơi chứa InsightFace models (tự tải về)
├── .env               # Cấu hình RTSP, Threshold, Model
├── .gitignore         # Loại bỏ file thừa
└── requirements.txt   # Thư viện cần thiết
```

## 🛠 Hướng dẫn Setup (Khuyên dùng Python 3.10)

### 1. Tạo môi trường ảo
Mở terminal tại thư mục gốc và chạy:
```bash
# Tạo venv với Python 3.10
py -3.10 -m venv venv

# Kích hoạt venv
venv\Scripts\activate

# Nâng cấp pip
pip install --upgrade pip
```

### 2. Cài đặt thư viện (Đặc biệt cho Windows)
Để tránh lỗi biên dịch C++, cài đặt theo thứ tự sau:
```bash
# 1. Cài đặt ONNX Runtime (CPU)
pip install onnxruntime==1.19.2

# 2. Cài đặt InsightFace bản Build sẵn (dành cho Python 3.10)
pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl

# 3. Cài đặt các thư viện còn lại
pip install -r requirements.txt
```

### 3. Cấu hình Camera
Copy file `.env.example` thành `.env` và cập nhật thông tin Camera:
```ini
RTSP_URL=rtsp://admin:your_password@192.168.1.100:554/h264/ch1/main/av_stream
INSIGHTFACE_MODEL=buffalo_l  # Hoặc buffalo_s để chạy nhanh hơn trên CPU yếu
RECOGNITION_THRESHOLD=0.4     # Ngưỡng nhận diện (0.35 - 0.45 là tối ưu)
```

## 🖥 Cách chạy
Hãy đảm bảo bạn đã kết nối Camera IP hoặc có luồng RTSP hợp lệ.

```bash
python -m src.main
python -m src.api
```

### Các phím điều khiển:
- `q`: Thoát chương trình.
- `e`: Đăng ký khuôn mặt mới (Enrollment). Nhập tên trong terminal sau khi bấm.

## 📈 Roadmap phát triển
1. **Database chuyên nghiệp**: Thay thế file `.pkl` bằng SQL (PostgreSQL) hoặc NoSQL (MongoDB).
2. **Web Dashboard**: Xây dựng UI quản lý nhân viên và xem lịch sử chấm công (FastAPI/React).
3. **Anti-spoofing**: Tích hợp module chống giả mạo khuôn mặt (Silent-Face-Anti-Spoofing).
4. **Edge Computing**: Tối ưu hóa chạy trên Jetson Nano hoặc Coral TPU.

## 📝 Lưu ý bảo mật
- Không chia sẻ file `.env` chứa mật khẩu RTSP.
- Đảm bảo tuân thủ các quy định về quyền riêng tư khi thu thập dữ liệu sinh trắc học.
