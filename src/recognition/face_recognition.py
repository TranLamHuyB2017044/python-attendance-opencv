"""
Face Recognition module using InsightFace.
Handles face detection and feature extraction (embedding).
"""

import numpy as np
import cv2
from insightface.app import FaceAnalysis
from loguru import logger
from typing import List, Optional, Tuple, Any

from src.config import InsightFaceConfig, MODELS_DIR, RecognitionConfig


class FaceRecognition:
    """
    Face Recognition handler using InsightFace (ArcFace).
    
    This class wraps InsightFace's FaceAnalysis to detect faces
    and generate 512-dimensional embeddings.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        det_size: Optional[Tuple[int, int]] = None,
        det_thresh: Optional[float] = None,
        ctx_id: int = -1,  # -1 for CPU, 0+ for GPU
    ):
        """
        Initialize InsightFace models.
        
        Args:
            model_name: Name of the model pack (e.g., 'buffalo_l', 'buffalo_s')
            det_size: Input size for face detection
            det_thresh: Threshold for face detection
            ctx_id: Context ID for execution (CPU/GPU)
        """
        self.model_name = model_name or InsightFaceConfig.MODEL_NAME
        self.det_size = det_size or InsightFaceConfig.DET_SIZE
        self.det_thresh = det_thresh or InsightFaceConfig.DET_THRESH
        
        try:
            # Initialize FaceAnalysis — detection + landmark only
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(MODELS_DIR),
                allowed_modules=['detection', 'landmark_2d_106'],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
            logger.info(f"InsightFace detection model ready. DET_SIZE={self.det_size}")

            # Manual load of recognition (ArcFace embedding) model
            from insightface.model_zoo import get_model
            rec_model_name = "w600k_r50.onnx" if "buffalo_l" in self.model_name else "w600k_mbf.onnx"
            rec_model_path = MODELS_DIR / "models" / self.model_name / rec_model_name
            self.rec_model = get_model(str(rec_model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.rec_model.prepare(ctx_id=ctx_id)
            logger.info("ArcFace recognition model loaded.")

            # Anti-Spoofing — chỉ load khi ANTI_SPOOFING_ENABLED=true
            if RecognitionConfig.ANTI_SPOOFING_ENABLED:
                from src.recognition.antispoofing import AntiSpoofing
                anti_spoof_dir = str(MODELS_DIR / "anti_spoof")
                self.anti_spoof = AntiSpoofing(model_dir=anti_spoof_dir)
                logger.info(f"[Anti-Spoofing] MiniFASNet loaded (ENABLED) from: {anti_spoof_dir}")
            else:
                self.anti_spoof = None
                logger.warning("[Anti-Spoofing] DISABLED — skipping model load. All faces treated as REAL.")

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise e

    def detect_and_extract(self, frame: np.ndarray, max_faces: Optional[int] = None,
                           fast: bool = False, roi=None) -> List[Any]:
        """
        Architecture:
        1. [Optional ROI crop] - chỉ detect trên vùng quan tâm → nhanh hơn
        2. InsightFace Detection (SCRFD) với cải tiến multi-angle
        3. InsightFace Recognition (ArcFace) — defer to tracker nếu fast=True

        roi: (x1, y1, x2, y2) tọa độ full frame.
             Nếu set → detect chỉ trên vùng đó, sau đó remap bbox về tọa độ gốc.
        """
        try:
            # ─── ROI Crop: giảm pixels → SCRFD nhanh hơn ──────────────────
            offset_x, offset_y = 0, 0
            detect_frame = frame

            if roi is not None:
                fx1, fy1, fx2, fy2 = roi
                fh, fw = frame.shape[:2]
                fx1 = max(0, fx1); fy1 = max(0, fy1)
                fx2 = min(fw, fx2); fy2 = min(fh, fy2)
                if fx2 > fx1 and fy2 > fy1:
                    detect_frame = frame[fy1:fy2, fx1:fx2]
                    offset_x, offset_y = fx1, fy1

            # 1. Detection & Landmarks với cải tiến multi-angle
            # Sử dụng detector với confidence threshold thấp hơn để detect mặt từ nhiều góc
            faces = self.app.get(detect_frame)

            # Remap bbox về tọa độ full frame nếu dùng ROI
            if offset_x != 0 or offset_y != 0:
                for face in faces:
                    face.bbox[0] += offset_x; face.bbox[2] += offset_x
                    face.bbox[1] += offset_y; face.bbox[3] += offset_y
                    if hasattr(face, 'kps') and face.kps is not None:
                        face.kps[:, 0] += offset_x
                        face.kps[:, 1] += offset_y

            if not faces:
                return []

            # Filter out faces that are too small or too far away
            valid_faces = []
            min_size = getattr(RecognitionConfig, "MIN_FACE_SIZE", 60)  # Giảm kích thước tối thiểu
            for face in faces:
                w = face.bbox[2] - face.bbox[0]
                h = face.bbox[3] - face.bbox[1]
                if w >= min_size and h >= min_size:
                    valid_faces.append(face)
            faces = valid_faces
            
            if not faces:
                return []

            # Sort by size
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            if max_faces is not None:
                faces = faces[:max_faces]

            if fast:
                # In fast mode, we just return the detections with placeholders
                for face in faces:
                    face.is_real = True # Default to True so it enters tracker's stabilizing
                    face.as_label = 2    # 2 = WAITING for analysis
                    face.as_score = 0.0
                return faces

            real_faces = []
            for face in faces:
                # --- TỐI ƯU HÓA: Cắt lấy vùng mặt kèm margin thay vì đưa cả frame lớn vào model ---
                x1, y1, x2, y2 = face.bbox.astype(int)
                h_b, w_b = y2 - y1, x2 - x1
                margin_x = int(w_b * 0.5)
                margin_y = int(h_b * 0.5)
                
                c_x1 = max(0, x1 - margin_x)
                c_y1 = max(0, y1 - margin_y)
                c_x2 = min(frame.shape[1], x2 + margin_x)
                c_y2 = min(frame.shape[0], y2 + margin_y)
                
                face_crop = frame[c_y1:c_y2, c_x1:c_x2].copy()
                
                # Backup toạ độ
                orig_bbox = face.bbox.copy()
                orig_kps = face.kps.copy() if face.kps is not None else None
                
                # Tịnh tiến về gốc toạ độ của ảnh đã cắt
                face.bbox[0] -= c_x1
                face.bbox[1] -= c_y1
                face.bbox[2] -= c_x1
                face.bbox[3] -= c_y1
                
                if face.kps is not None:
                    face.kps[:, 0] -= c_x1
                    face.kps[:, 1] -= c_y1

                # 2. Anti-Spoofing — chỉ chạy nếu được bật VÀ model đã load
                if self.anti_spoof is not None and RecognitionConfig.ANTI_SPOOFING_ENABLED:
                    as_label, as_score = self.anti_spoof.predict(face_crop, face)
                    face.as_label = int(as_label)
                    face.as_score = float(as_score)
                    face.is_real = (face.as_label == 1)
                else:
                    # Anti-spoofing tắt: mặc định là người thật, as_label=1
                    face.is_real = True
                    face.as_label = 1
                    face.as_score = 1.0

                # 3. Face Alignment for better recognition from top-down angles
                if face.is_real and hasattr(face, 'kps') and face.kps is not None:
                    # Áp dụng face alignment để cải thiện recognition với góc từ trên xuống
                    aligned_face = self._align_face_for_recognition(face_crop, face.kps)
                    if aligned_face is not None:
                        face_crop = aligned_face

                # 4. Recognition (ArcFace embedding)
                if face.is_real:
                    self.rec_model.get(face_crop, face)
                    
                # Khôi phục toạ độ nguyên vẹn để trả về cho hệ thống
                face.bbox = orig_bbox
                if orig_kps is not None:
                    face.kps = orig_kps

                real_faces.append(face)

            return real_faces
            
        except Exception as e:
            logger.error(f"Error during face detection/extraction: {e}")
            return []

    def _align_face_for_recognition(self, face_crop: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Áp dụng face alignment để cải thiện recognition với góc từ trên xuống và các góc không trực diện.
        
        Args:
            face_crop: Ảnh mặt đã cắt
            landmarks: 5 facial landmarks (mắt, mũi, miệng)
            
        Returns:
            Aligned face image or None nếu alignment fails
        """
        try:
            # Landmarks format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
            # [0,1]: left eye, right eye; [2]: nose; [3,4]: mouth corners
            
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Tính góc nghiêng của mặt dựa trên vị trí mắt
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Nếu góc nghiêng lớn (mặt nghiêng nhiều), áp dụng rotation
            if abs(angle) > 10:  # Ngưỡng góc nghiêng
                center = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned_face = cv2.warpAffine(face_crop, rotation_matrix, 
                                            (face_crop.shape[1], face_crop.shape[0]),
                                            flags=cv2.INTER_CUBIC)
                return aligned_face
            
            # Nếu mặt thẳng hoặc nghiêng ít, không cần alignment
            return face_crop
            
        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return face_crop

    def draw_faces(self, frame: np.ndarray, faces: List[Any]) -> np.ndarray:
        """
        Draw bounding boxes and detailed metadata with plain text (no shadow).
        """
        res_frame = frame.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            display_name = getattr(face, 'name', '') or ''

            is_recognized = getattr(face, 'recognized', False)
            is_spoof      = getattr(face, 'is_spoof',   False)

            if is_spoof or any(kw in str(display_name) for kw in ["TRUY CAP", "GIA MAO"]):
                color = (0, 0, 255)       # 🔴 Đỏ: spoof / unauthorized
            elif is_recognized:
                color = (0, 220, 0)       # 🟢 Xanh: đã nhận diện thành công
            else:
                color = (0, 200, 255)     # 🟡 Vàng: đang phân tích / chưa rõ


            # Box
            cv2.rectangle(res_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            y_offset = bbox[1] - 10

            if hasattr(face, 'name'):
                score = getattr(face, 'score', 0.0) or 0.0
                label = f"{face.name} ({score:.2f})" if is_recognized and score > 0 else face.name
                cv2.putText(res_frame, label, (bbox[0], y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                from src.config import CameraConfig
                import time
                cam_time_label = f"{CameraConfig.CAMERA_NAME} | {time.strftime('%H:%M:%S')}"
                cv2.putText(res_frame, cam_time_label, (bbox[0], y_offset - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                if is_recognized:
                    meta_info = [
                        f"ID: {getattr(face, 'user_id', 'Unknown') or 'Unknown'}",
                        f"N-sinh: {getattr(face, 'birthday', 'N/A') or 'N/A'}",
                        f"Gio: {getattr(face, 'detect_time', 'N/A') or 'N/A'}",
                        f"Mau: {getattr(face, 'vector_count', 0) if getattr(face, 'vector_count', None) is not None else 0}"
                    ]
                    for i, text in enumerate(meta_info):
                        cv2.putText(res_frame, text, (bbox[0], bbox[3] + 25 + i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            # If SPOOF or UNAUTHORIZED, add a big warning
            if color == (0, 0, 255):
                warning_msg = "WARNING: SECURITY ALERT!"
                if "CANH BAO" in display_name or "SPOOF" in display_name:
                    warning_msg = "WARNING: ANTI-SPOOFING TRIGGERED!"
                elif "TRUY CAP" in display_name:
                    warning_msg = "WARNING: UNAUTHORIZED ACCESS!"
                    
                cv2.putText(res_frame, warning_msg, (bbox[0], bbox[3] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        return res_frame
                
        return res_frame


__all__ = ["FaceRecognition"]
