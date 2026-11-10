"""SQLite queue for camera events awaiting an acknowledgement from FastAPI."""
import json
import sqlite3
from contextlib import closing
from threading import Event, Thread

from loguru import logger

from src.attendance.device_credentials import camera_data_dir, protect, unprotect
from src.attendance_transport import decode_value, encode_value


class CameraOutbox:
    def __init__(self, device_id, send, acknowledged):
        self.device_id = device_id
        self.send = send
        self.acknowledged = acknowledged
        directory = camera_data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "attendance-outbox.sqlite3"
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS events (
                       id INTEGER PRIMARY KEY, device_id TEXT NOT NULL, payload BLOB NOT NULL,
                       rejected INTEGER, last_error INTEGER
                   )"""
            )
        self.wake = Event()
        self.worker = Thread(target=self.run, name="attendance-outbox", daemon=True)

    def start(self):
        self.worker.start()

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def put(self, arguments, event):
        payload = protect(json.dumps([arguments, event], default=encode_value).encode("utf-8"))
        with closing(self.connect()) as connection, connection:
            connection.execute("INSERT INTO events (device_id, payload) VALUES (?, ?)", (self.device_id, payload))
        self.wake.set()

    def run(self):
        delay = 1
        while True:
            try:
                drained = self.drain_one()
            except Exception as error:
                logger.warning("Camera queue could not be processed ({})", type(error).__name__)
                drained = False
            if drained:
                delay = 1
                continue
            self.wake.wait(delay)
            self.wake.clear()
            delay = min(delay * 2, 30)

    def drain_one(self):
        with closing(self.connect()) as connection:
            item = connection.execute(
                "SELECT id, payload FROM events WHERE device_id = ? AND rejected IS NULL ORDER BY id LIMIT 1",
                (self.device_id,),
            ).fetchone()
        if not item:
            return False
        identifier, payload = item
        arguments, event = json.loads(unprotect(payload), object_hook=decode_value)
        try:
            status, log_entry = self.send("log_attendance", *arguments, **event)
        except Exception as error:
            code = getattr(error, "status_code", None)
            if code in {400, 403, 404, 413, 422}:
                with closing(self.connect()) as connection, connection:
                    connection.execute("UPDATE events SET rejected = 1, last_error = ? WHERE id = ?", (code, identifier))
                logger.error("Camera event {} requires review (HTTP {})", event["event_id"], code)
                return True
            # Authentication and transient errors keep events available for retry.
            raise
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM events WHERE id = ?", (identifier,))
        if log_entry:
            self.acknowledged(log_entry, arguments[3])
        return True
