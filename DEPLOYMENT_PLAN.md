# Kế hoạch Đóng gói và Cập nhật Ứng dụng (Loader + Source Flow)

Tài liệu này hướng dẫn chi tiết cách triển khai mô hình đóng gói mới cho hệ thống chấm công, cho phép cập nhật tính năng (như Telegram Bot) mà không cần build lại file EXE 500MB.

## 1. Cấu trúc Thư mục Phân phối (Distribution)
Sau khi chạy script `build_dual_apps.py`, thư mục gửi cho khách hàng sẽ có cấu trúc như sau:

```text
AttendanceSystem/
├── _internal/            # CHỨA RUNTIME (Nặng, không thay đổi)
│   ├── python.exe        # Trình thông dịch Python đóng gói
│   ├── cv2/              # Thư viện OpenCV
│   ├── torch/            # Thư viện AI
│   └── ...               # Các file DLL và thư viện khác
├── src/                  # CHỨA LOGIC (Nhẹ, cập nhật thường xuyên)
│   ├── recognition/      # tracker.py, face_recognition.py...
│   ├── utils/            # telegram_bot.py, logger.py...
│   └── ...               # Các file .py khác
├── AttendanceService.exe # Loader cho Camera Service (chạy nền)
├── AttendanceManager.exe # Loader cho Attendance Manager (giao diện)
├── .env                  # File cấu hình (Token Telegram, API Key...)
└── models/               # Các file model AI (.onnx)
```

---

## 2. Quy trình Thực hiện (Flow)

### Bước 1: Đóng gói lần đầu (Lần đầu cài cho khách)
1. Chạy script build: `python build_dual_apps.py`.
2. Lấy thư mục `dist_package/AttendanceSystem` nén thành file ZIP gửi cho khách hàng.

### Bước 2: Khi cần cập nhật tính năng mới (Ví dụ: Sửa Telegram Bot)
Thay vì gửi lại 500MB, bạn thực hiện:
1. Sửa code trong file `src/utils/telegram_bot.py` trên máy của bạn.
2. Gửi duy nhất file `telegram_bot.py` đó cho khách hàng qua Zalo/Email (chỉ vài KB).
3. Hướng dẫn khách hàng copy file đó và ghi đè vào thư mục `src/utils/` trong máy họ.
4. Yêu cầu khách Restart lại `AttendanceService` (hoặc khởi động lại máy) để áp dụng.

---

## 3. Ưu điểm của mô hình này

| Đặc điểm | Build EXE truyền thống | Mô hình Loader + Source |
| :--- | :--- | :--- |
| **Dung lượng cập nhật** | ~500 MB (Toàn bộ EXE) | ~10 KB (Chỉ file .py thay đổi) |
| **Thời gian cập nhật** | 5 - 10 phút (Tải + Cài lại) | < 5 giây (Copy - Paste) |
| **Môi trường khách hàng** | Không cần Python | Không cần Python (Dùng _internal) |
| **Độ ổn định** | Cao | Rất cao (Không chạm vào Runtime) |
| **Quản lý 2 App** | Phải build 2 lần riêng biệt | Dùng chung 1 Runtime, cập nhật 1 lần |

---

## 4. Các lưu ý quan trọng

1. **Bảo mật mã nguồn**: Nếu bạn không muốn khách hàng đọc được code trong thư mục `src`, hãy biên dịch chúng sang file `.pyc` (Python Compiled) trước khi gửi. Loader vẫn có thể đọc được file `.pyc` bình thường.
2. **Cập nhật thư viện**: Nếu tính năng mới yêu cầu cài thêm thư viện (ví dụ: `pip install new-library`), lúc đó bạn mới cần build lại toàn bộ thư mục `_internal`. Tuy nhiên, các thư viện lõi như OpenCV, Torch rất ít khi thay đổi.
3. **Cấu hình động**: Luôn ưu tiên đưa các biến cấu hình (Token, ID) vào file `.env` hoặc Database để có thể thay đổi "nóng" mà không cần gửi lại file code.

---
*Tài liệu được tạo tự động bởi AI Assistant phục vụ dự án Bittech Attendance.*
