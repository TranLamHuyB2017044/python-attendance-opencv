import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.config import EmailConfig
from loguru import logger

def send_reset_password_email(receiver_email, username, new_password):
    """
    Sends an email with the new reset password to the user.
    """
    if not EmailConfig.SMTP_USERNAME or not EmailConfig.SMTP_PASSWORD:
        logger.error("SMTP Email not configured. Please set SMTP_USERNAME and SMTP_PASSWORD in .env")
        return False, "Hệ thống chưa cấu hình Email gửi mật khẩu."

    try:
        msg = MIMEMultipart()
        msg['From'] = f"BITTECH AI System <{EmailConfig.SMTP_USERNAME}>"
        msg['To'] = receiver_email
        msg['Subject'] = "Khôi phục mật khẩu hệ thống BITTECH AI"

        body = f"""
        Chào {username},

        Bạn (hoặc ai đó) đã yêu cầu khôi phục mật khẩu cho tài khoản '{username}' trên hệ thống BITTECH AI.

        Mật khẩu mới của bạn là: {new_password}

        Vui lòng đăng nhập và đổi lại mật khẩu ngay lập tức để đảm bảo bảo mật.

        Trân trọng,
        Đội ngũ BITTECH AI
        """
        msg.attach(MIMEText(body, 'plain'))

        # Create server connection
        # Port 465 is for SSL, Port 587 is for TLS
        if EmailConfig.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(EmailConfig.SMTP_HOST, EmailConfig.SMTP_PORT)
        else:
            server = smtplib.SMTP(EmailConfig.SMTP_HOST, EmailConfig.SMTP_PORT)
            server.starttls()
        
        server.login(EmailConfig.SMTP_USERNAME, EmailConfig.SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(EmailConfig.SMTP_USERNAME, receiver_email, text)
        server.quit()

        logger.info(f"Successfully sent reset password email to {receiver_email}")
        return True, "Mật khẩu mới đã được gửi vào email của bạn."
    except Exception as e:
        logger.error(f"Failed to send email to {receiver_email}: {e}")
        return False, f"Lỗi gửi email: {str(e)}"
