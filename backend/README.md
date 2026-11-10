# FastAPI cho dữ liệu chấm công

Ứng dụng quản lý dùng tài khoản hiện có để đăng nhập. Mỗi máy camera có một API key riêng và chạy độc lập với phiên đăng nhập. Quản trị viên có thể cấp quyền chấm công cho nhiều công ty trên cùng một camera. Backend là nơi duy nhất dùng PyMongo.

## Backend

Tạo `backend/.env` từ [mẫu cấu hình](.env.example). Đặt URI và tên MongoDB hiện có, sau đó tạo một khóa ký JWT riêng cho backend. Cấu hình `ATTENDANCE_PUBLIC_URL` bằng địa chỉ HTTPS mà các bên cần xem ảnh chấm công có thể truy cập.

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -m pip install -r backend/requirements.txt
python backend/main.py
```

Mặc định FastAPI nghe trên `127.0.0.1:8001`. Dùng HTTPS khi truy cập từ máy khác, ví dụ qua reverse proxy có cấu hình giới hạn kích thước request và không ghi token trong query string vào access log. Kiểm tra `/health/live` và `/health/ready`. Readiness khởi tạo các index nếu cần.

## Bản cài Windows

1. Cấu hình `ATTENDANCE_BACKEND_URL` trên máy khách. Build lại bằng `python build_dual_apps.py`. Giữ khóa ký JWT, URI MongoDB và các khóa tích hợp bên ngoài bản phát hành.
2. Đăng nhập AttendanceManager bằng tài khoản admin. Mở **Thiết bị**, tạo một key cho từng máy chấm công. Chọn công ty dùng cấu hình RTSP, tích thêm các công ty được chấm công và điền thông tin HKB nếu cần. Key chỉ xuất hiện tại lúc tạo hoặc đổi. Thiết bị đã có key có thể cập nhật danh sách bằng nút **Công ty chấm công** và tiếp tục dùng key hiện tại. Các thiết bị cũ mặc định chỉ được phép chấm công cho công ty cấu hình.
3. Trên máy camera, đăng nhập AttendanceManager và mở **Cấu hình Camera Service** để dán key. Mở app bằng cùng tài khoản Windows sẽ chạy AttendanceService. Key và cấu hình được bảo vệ bằng DPAPI trong `%LOCALAPPDATA%\Bittech\Attendance`. Để chạy tự động khi đăng nhập Windows, cấu hình Task Scheduler chạy AttendanceService.exe dưới cùng tài khoản, với thư mục làm việc là thư mục cài đặt.
4. Sau khi xác nhận heartbeat và bản ghi chấm công, đăng xuất app quản lý và kiểm tra camera tiếp tục hoạt động. Khi đổi key cho cùng thiết bị, key trước đó còn hiệu lực trong 15 phút. Thu hồi thiết bị sẽ vô hiệu hóa cả hai key.

Các tài khoản và bản ghi chấm công cũ vẫn được sử dụng. Trước khi chuyển một máy, kiểm tra công ty của hồ sơ nhân viên và giấy phép HKB. Chuyển thông tin RTSP dùng riêng từ cấu hình GLOBAL sang cấu hình công ty tương ứng. Tài khoản công ty chỉ được xem thêm công ty có trong `users.allowed_company_ids` do quản trị hệ thống gán.

## Qdrant, mất mạng và HKB

Dùng Qdrant riêng theo công ty, hoặc một môi trường Qdrant đã cô lập quyền truy cập theo công ty. Bản app hiện vẫn kết nối Qdrant trực tiếp. Khóa Qdrant có quyền toàn bộ kho dữ liệu dùng chung sẽ cấp quyền đó cho máy giữ khóa.

Camera tìm kiếm trong các công ty được cấp quyền và ghi chấm công theo công ty của hồ sơ khớp. Các vector của những công ty dùng chung camera cần có trong Qdrant mà máy truy cập. Camera xếp hàng sự kiện đã nhận diện trong SQLite, bảo vệ nội dung bằng DPAPI và gửi lại khi FastAPI phục hồi. Mỗi sự kiện giữ nguyên ID và thời điểm ghi nhận. Các sự kiện bị từ chối do dữ liệu hoặc do công ty đã bị gỡ quyền được giữ lại để kiểm tra. Hàng đợi hoạt động khi RTSP và Qdrant vẫn truy cập được.

Ứng dụng tiếp tục gọi trực tiếp API HKB qua luồng đồng bộ hiện tại sau khi FastAPI xác nhận bản ghi. Camera lấy cấu hình HKB và cập nhật kết quả qua FastAPI trong phạm vi thiết bị. Giữ thông tin xác thực HKB hiện có và cấu hình lại phạm vi kết nối ở bước cấp key.

## API tra cứu cũ và chuyển đổi

API tra cứu cũ có thể chạy riêng với token tích hợp chỉ đọc và một công ty cụ thể. Đặt `ATTENDANCE_INTEGRATION_TOKEN` và `ATTENDANCE_INTEGRATION_COMPANY` trên backend và tiến trình API tra cứu. Đặt thêm `ATTENDANCE_EXTERNAL_API_TOKEN` trên tiến trình API tra cứu để xác thực bên gọi ngoài. Endpoint ảnh tra cứu yêu cầu token. Link ảnh dashboard được FastAPI ký và tự hết hạn. Đăng ký khuôn mặt qua AttendanceManager hoặc route Flask `/enroll` với JWT quản lý.

Thử trên một máy trước, sau đó triển khai từng nhóm máy. Backend mới yêu cầu bản app có JWT hoặc API key thiết bị, vì vậy hãy lên lịch chuyển backend và client trong cùng đợt. Nếu cần chạy thử song song, dùng một backend mới trên địa chỉ riêng và chuyển URL từng nhóm máy. Khi triển khai xong, xóa cấu hình token dùng chung cũ khỏi máy khách.
 
## Kiểm thử cục bộ

```powershell
python -m pip install mongomock httpx
python -m unittest discover -s backend/tests -v
```
