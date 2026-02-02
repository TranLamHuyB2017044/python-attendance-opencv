import ntplib
import time
import pytz
from datetime import datetime
from loguru import logger

class TimeManager:
    """
    Ensures accurate time even if the system clock is manipulated.
    Always returns time in Vietnam Timezone (UTC+7).
    """
    def __init__(self):
        self.tz = pytz.timezone('Asia/Ho_Chi_Minh')

    def get_accurate_time(self):
        """
        Fetches time from NTP servers and converts to Vietnam timezone.
        Returns a localized datetime object.
        """
        ntp_servers = ['pool.ntp.org', 'time.google.com', 'time.windows.com']
        client = ntplib.NTPClient()
        
        for server in ntp_servers:
            try:
                response = client.request(server, timeout=2)
                # Convert UTC timestamp to localized Vietnam time
                utc_dt = datetime.fromtimestamp(response.tx_time, pytz.utc)
                vn_dt = utc_dt.astimezone(self.tz)
                logger.debug(f"Time synced with {server} (VN Time)")
                return vn_dt
            except Exception:
                continue
        
        # Fallback to system time converted to VN timezone if internet is down
        logger.warning("Could not sync with NTP servers. Using system clock (converted to VN Time).")
        now_utc = datetime.now(pytz.utc)
        return now_utc.astimezone(self.tz)

    def get_formatted_time(self):
        """Returns (timestamp_str, date_str) in VN Timezone"""
        now = self.get_accurate_time()
        return now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d")

# Global access
time_mgr = TimeManager()
