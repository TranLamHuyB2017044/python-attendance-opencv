import asyncio
from bittech_auth import HKBClient
from src.config import AuthServiceConfig
from loguru import logger
import os
import json

class HKBService:
    """
    Adapter service for HKB Auth SDK.
    Provides synchronous methods to interact with the HKB Auth system.
    """
    def __init__(self):
        self.client = HKBClient(
            base_url=AuthServiceConfig.BASE_URL
        )
        self.loop = asyncio.new_event_loop()

    def _run_sync(self, coro):
        """Helper to run async code in a synchronous manner."""
        try:
            return self.loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"HKB Service: Sync execution error: {e}")
            return None

    def register_client(self, system_id, external_id, description, user_info, system_connection_id=1, system_register="face_recognition"):
        """
        Register a new client system.
        """
        try:
            logger.info(f"HKB Service: Registering client {system_id} for external_id {external_id}")
            coro = self.client.register_client(
                system_connection_id=system_connection_id,
                system_id=system_id,
                system_register=system_register,
                external_id=external_id,
                description=description,
                user_info=user_info
            )
            result = self._run_sync(coro)
            return result
        except Exception as e:
            logger.error(f"HKB Service: Register client failed: {e}")
            return None

    def authenticate(self, system_id, api_key, user_id):
        """
        Authenticate a user.
        """
        try:
            logger.info(f"HKB Service: Authenticating user {user_id} for system {system_id}")
            coro = self.client.authenticate(
                system_id=system_id,
                api_key=api_key,
                user_id=user_id
            )
            auth = self._run_sync(coro)
            return auth
        except Exception as e:
            logger.error(f"HKB Service: Authenticate failed: {e}")
            return None

    def get_connections(self):
        """
        Get list of connections for the current API Key.
        """
        try:
            logger.info("HKB Service: Fetching connections list")
            coro = self.client.get_system_connections(
                group_key="dwX1S5cHAPDYo6Gom2fv8F3D7rNZqPu",
                client_registers=[""]
            )
            result = self._run_sync(coro)
            
            if result:
                logger.info(f"HKB Service Response: Success={result.success}, Message={result.message}, Data={result.data}")
            else:
                logger.warning("HKB Service Response: None (Sync execution failed or returned invalid data)")

            if result and result.success:
                return result.data
            return None
        except Exception as e:
            logger.error(f"HKB Service: Get connections failed: {e}")
            return None

    def upload_attendance(self, user_id, user_name, timestamp, status, image_path=None):
        """
        Upload attendance log to HKB service by sharing it to an external endpoint.
        """
        try:
            logger.info(f"HKB Service: Uploading attendance for {user_name} ({user_id})")
            
            # Since the current SDK uses upload_document and expects an endpoint,
            # we might need to find which connection to use.
            connections = self.get_connections()
            if not connections or not isinstance(connections, list) or len(connections) == 0:
                logger.warning("HKB Service: No connections found to upload to.")
                return None
            
            # Use the first connection for demonstration
            conn = connections[0]
            endpoint = conn.get("endpoint")
            if not endpoint:
                logger.warning("HKB Service: Connection has no endpoint.")
                return None

            payload = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp,
                "status": status,
                "app_name": "Face Attendance System"
            }
            
            # If there's an image, we might need a different method or encode it
            # The current upload_document in SDK only takes a dict payload.
            # Usually images are uploaded as Multipart, but let's stick to what's in SDK.
            
            coro = self.client.upload_document(
                endpoint=endpoint,
                system_id=AuthServiceConfig.SYSTEM_ID,
                api_key=AuthServiceConfig.API_KEY,
                user_id=1, # Default user_id for system auth
                payload=payload
            )
            result = self._run_sync(coro)
            return result
        except Exception as e:
            logger.error(f"HKB Service: Upload attendance failed: {e}")
            return None

# Global instance
hkb_service = HKBService()
