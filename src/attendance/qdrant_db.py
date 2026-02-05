from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger
import uuid

from src.config import QdrantConfig, RecognitionConfig

class QdrantAttendanceManager:
    """
    Manager for face embeddings using Qdrant Vector Database.
    """
    
    def __init__(self):
        """Initialize Qdrant client and ensure collection exists."""
        self.client = QdrantClient(
            url=QdrantConfig.URL,
            api_key=QdrantConfig.API_KEY,
        )
        self.collection_name = QdrantConfig.COLLECTION_NAME
        self.threshold = RecognitionConfig.THRESHOLD
        self.vector_size = RecognitionConfig.EMBEDDING_DIM
        
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection and indexes if they don't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.success(f"Collection '{self.collection_name}' created.")
            
            # Ensure payload index for user_id exists (required for filtering in count)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Payload index for 'user_id' ensured in {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection/indexes: {e}")
            raise e

    def upsert_user(self, user_name: str, user_id: str, birthday: str, embeddings: List[np.ndarray], **kwargs) -> bool:
        """
        Save/Update user embeddings in Qdrant with metadata.
        """
        try:
            points = []
            for emb in embeddings:
                vector = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "user_name": user_name,
                        "birthday": birthday,
                        "company_id": kwargs.get("company_id", "default"),
                        "created_at": datetime.now().isoformat()
                    }
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.success(f"Upserted {len(points)} samples for {user_name} (ID: {user_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert user {user_name}: {e}")
            return False

    def recognize(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Search for the closest face and return detailed user info.
        """
        default_res = {
            "name": "Unknown",
            "user_id": "Unknown",
            "birthday": "N/A",
            "score": 0.0,
            "detect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vector_count": 0
        }
        
        try:
            vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=1,
                with_payload=True
            )

            if not search_result or not search_result.points:
                return default_res

            result = search_result.points[0]
            score = float(result.score)
            
            if score < self.threshold:
                default_res["score"] = score
                return default_res

            payload = result.payload
            user_id = payload.get("user_id")

            # Count total vectors for this user
            count_res = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                )
            )

            return {
                "name": payload.get("user_name", "Unknown"),
                "user_id": user_id,
                "birthday": payload.get("birthday", "N/A"),
                "score": score,
                "detect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vector_count": count_res.count
            }

        except Exception as e:
            logger.error(f"Error during recognition: {e}")
            return default_res

    def get_all_users(self, company_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves unique users. Optional filtering by company_id.
        """
        try:
            all_users = {}
            offset = None
            
            scroll_filter = None
            if company_id:
                scroll_filter = models.Filter(
                    must=[models.FieldCondition(key="company_id", match=models.MatchValue(value=company_id))]
                )
            
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    offset=offset
                )
                
                for point in results:
                    payload = point.payload
                    u_id = payload.get("user_id")
                    if u_id and u_id not in all_users:
                        all_users[u_id] = {
                            "user_id": u_id,
                            "user_name": payload.get("user_name"),
                            "birthday": payload.get("birthday")
                        }
                
                if offset is None:
                    break
            
            return list(all_users.values())
        except Exception as e:
            logger.error(f"Error fetching all users: {e}")
            return []

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user info by user_id from Qdrant.
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                ),
                limit=1,
                with_payload=True
            )
            
            if results and results[0]:
                payload = results[0][0].payload
                return {
                    "user_id": payload.get("user_id"),
                    "user_name": payload.get("user_name"),
                    "birthday": payload.get("birthday")
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching user info for {user_id}: {e}")
            return None

    def update_user_info(self, user_id: str, new_name: str, new_birthday: str) -> bool:
        """
        Update user_name and birthday for all points with given user_id.
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "user_name": new_name,
                    "birthday": new_birthday
                },
                points=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                )
            )
            logger.success(f"Updated info for user ID: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update user info for {user_id}: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """
        Delete all points associated with a user_id.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                    )
                )
            )
            logger.success(f"Deleted user ID: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

__all__ = ["QdrantAttendanceManager"]
