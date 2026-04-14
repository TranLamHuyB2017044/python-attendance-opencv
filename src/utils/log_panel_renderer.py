import cv2
import time
import datetime
import numpy as np
from loguru import logger

# ── Status colors / short labels ─────────────────────────────────────────────
_S_COLOR = {
    "IN":       (0,  210,  80),
    "OUT":      (50,  90, 240),
    "COOLDOWN": (0,  200, 240),
    "SPOOF":    (0,   50, 230),
    "unknown":  (100, 100, 100),
    "DETECTED": (160, 110,  0),
}
_S_LABEL = {
    "IN": "IN", "OUT": "OUT", "COOLDOWN": "CD",
    "SPOOF": "SP", "unknown": "??", "DETECTED": "DT",
}

def draw_log_panel(canvas: np.ndarray, x0: int, pw: int, ph: int):
    """
    Render right-side dual-stream log panel onto `canvas`.
    """
    F   = cv2.FONT_HERSHEY_SIMPLEX
    FB  = cv2.FONT_HERSHEY_DUPLEX
    PAD = 6

    # ── Background ───────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (x0, 0), (x0 + pw, ph), (5, 5, 5), -1)  # Nền đen cho panel
    cv2.rectangle(canvas, (x0, 0), (x0 + 2, ph), (80, 80, 80), -1) # Viền xám phân cách

    # ── Header bar ───────────────────────────────────────────────────────────
    HDR = 56
    cv2.rectangle(canvas, (x0, 0), (x0 + pw, HDR), (0, 0, 0), -1)

    cv2.putText(canvas, "WEBHOOK MONITOR",
                (x0 + PAD + 4, 20), FB, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    dot = (0, 230, 70) if int(time.time()) % 2 == 0 else (0, 90, 35)
    cv2.circle(canvas, (x0 + pw - 12, 16), 5, dot, -1)

    cy0, cy1 = 30, HDR - 6
    lx = x0 + PAD + 4
    cv2.rectangle(canvas, (lx, cy0), (lx + 54, cy1), (28, 120, 28), -1, cv2.LINE_AA)
    cv2.putText(canvas, "^ SENT", (lx + 3, cy1 - 2), F, 0.30, (255, 255, 255),  1, cv2.LINE_AA)
    
    lx2 = lx + 60
    cv2.rectangle(canvas, (lx2, cy0), (lx2 + 50, cy1), (200, 100, 30), -1, cv2.LINE_AA)
    cv2.putText(canvas, "v GET",  (lx2 + 3, cy1 - 2), F, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, datetime.datetime.now().strftime("%H:%M:%S"),
                (x0 + pw - 62, cy1 - 2), F, 0.31, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Pull entries ──────────────────────────────────────────────────────────
    try:
        from src.utils.webhook_log_bus import get_entries, mark_all_read
        entries = get_entries(limit=50)
        mark_all_read()
    except Exception:
        entries = []

    # ── Rows ──────────────────────────────────────────────────────────────────
    ROW_H = 50
    y     = HDR + 4

    for ent in entries:
        if y + ROW_H > ph - 22:
            break

        log_type = ent.get("log_type", "GET")
        status   = ent.get("status",   "unknown")
        name     = ent.get("user_name","Unknown")
        uid      = str(ent.get("user_id", ""))
        t_str    = ent.get("time", "")
        is_new   = ent.get("is_new", False)

        # Nền đen cho mỗi log, hơi sáng lên khi có log mới
        bg = (30, 30, 30) if is_new else (0, 0, 0)
        cv2.rectangle(canvas, (x0 + 4, y), (x0 + pw - 4, y + ROW_H - 2), bg, -1)

        # Dải màu trái phân biệt GET và SENT (BGR)
        bar = (40, 180, 40) if log_type == "SENT" else (220, 120, 40) # Xanh lá vs Xanh lam nhẹ
        cv2.rectangle(canvas, (x0 + 4, y), (x0 + 14, y + ROW_H - 2), bar, -1)
        arrow = "^" if log_type == "SENT" else "v"
        cv2.putText(canvas, arrow, (x0 + 5, y + ROW_H // 2 + 5),
                    F, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        sc  = _S_COLOR.get(status, (100, 100, 100))
        sl  = _S_LABEL.get(status, status[:2].upper())
        px, py_p, pw_p, ph_p = x0 + 17, y + 7, 28, 16
        cv2.rectangle(canvas, (px, py_p), (px + pw_p, py_p + ph_p), sc, -1, cv2.LINE_AA)
        cv2.putText(canvas, sl, (px + 2, py_p + 12),
                    F, 0.27, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, t_str, (px + pw_p + 4, y + 18),
                    F, 0.33, (180, 180, 180), 1, cv2.LINE_AA)

        # Tên người: Chữ Trắng
        nc        = (255, 255, 255)
        max_ch    = max(10, (pw - 30) // 7)
        name_disp = (name[:max_ch - 2] + "..") if len(name) > max_ch else name
        cv2.putText(canvas, name_disp, (x0 + 17, y + 37),
                    F, 0.38, nc, 1, cv2.LINE_AA)

        if uid and uid not in ("Unknown", ""):
            cv2.putText(canvas, f"#{uid[-8:]}",
                        (x0 + pw - 62, y + 37),
                        F, 0.27, (140, 140, 140), 1, cv2.LINE_AA)

        dc = (40, 40, 40)
        cv2.line(canvas, (x0 + 4, y + ROW_H - 1), (x0 + pw - 4, y + ROW_H - 1), dc, 1)
        y += ROW_H

    # ── Footer & Clear Button ────────────────────────────────────────────────
    cv2.rectangle(canvas, (x0, ph - 20), (x0 + pw, ph), (12, 12, 12), -1)
    
    # Draw CLEAR button region (will use for click detection)
    btn_x = x0 + pw - 55
    btn_y1 = ph - 17
    btn_y2 = ph - 3
    cv2.rectangle(canvas, (btn_x, btn_y1), (x0 + pw - 4, btn_y2), (60, 60, 60), -1, cv2.LINE_AA)
    cv2.putText(canvas, "CLEAR", (btn_x + 5, ph - 6), F, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
    
    sc_cnt = sum(1 for e in entries if e.get("log_type") == "SENT")
    gc_cnt = sum(1 for e in entries if e.get("log_type") == "GET")
    cv2.putText(canvas,
                f"^ {sc_cnt} sent   v {gc_cnt} get",
                (x0 + PAD, ph - 6),
                F, 0.30, (120, 140, 160), 1, cv2.LINE_AA)
