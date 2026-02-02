import pyttsx3
import threading
from loguru import logger

class VoiceManager:
    """
    Handles text-to-speech feedback in a non-blocking way.
    """
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            # Set Vietnamese voice if available, otherwise default
            voices = self.engine.getProperty('voices')
            found_vi = False
            for voice in voices:
                v_name = voice.name.lower()
                v_id = voice.id.lower()
                if any(kw in v_name or kw in v_id for kw in ["viet", "vi-vn", "vn-", "vn_", " an"]):
                    self.engine.setProperty('voice', voice.id)
                    logger.info(f"Voice Manager: Selected Vietnamese voice -> {voice.name}")
                    found_vi = True
                    break
            
            if not found_vi:
                logger.warning("Voice Manager: Vietnamese voice not found. Using default voice.")
            
            self.engine.setProperty('rate', 160) # Speed suitable for VN
        except Exception as e:
            logger.error(f"Failed to initialize Voice Manager: {e}")
            self.engine = None

    def _speak(self, text):
        if self.engine:
            try:
                # pyttsx3 needs a new engine instance for multi-threading in some OS
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                for voice in voices:
                    v_name = voice.name.lower()
                    v_id = voice.id.lower()
                    if any(kw in v_name or kw in v_id for kw in ["viet", "vi-vn", "vn-", "vn_", " an"]):
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS Error: {e}")

    def speak(self, text):
        """Play voice in a separate thread to avoid blocking the camera stream."""
        if self.engine:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

# Global instance
voice_mgr = VoiceManager()
