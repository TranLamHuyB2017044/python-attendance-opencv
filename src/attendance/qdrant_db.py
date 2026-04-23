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
            
            # Ensure payload index for user_id and company_id exists
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="company_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Payload indexes for 'user_id' and 'company_id' ensured in {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection/indexes: {e}")
            raise e

    def upsert_user(self, user_name: str, user_id: str, birthday: str, embeddings: List[np.ndarray], clear_old: bool = False, enrollment_image_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Save/Update user embeddings in Qdrant with metadata.
        """
        try:
            if not embeddings:
                logger.warning(f"Upsert: No embeddings provided for user {user_name} (ID: {user_id})")
                return False

            user_id_str = str(user_id)
            
            # Use provided company_id or fallback. Handle None/Empty/ALL cases.
            raw_cid = kwargs.get("company_id")
            if raw_cid in [None, "", "ALL"]:
                # If 'ALL' or empty is passed, we fallback to a default since points must have a specific owner
                cid = "default"
            else:
                cid = str(raw_cid)
            
            active = kwargs.get("active", True)
            
            # If specified, remove existing vectors for this user first
            if clear_old:
                self.delete_user(user_id_str)

            points = []
            for i, emb in enumerate(embeddings):
                vector = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                payload = {
                    "user_id": user_id_str,
                    "user_name": user_name,
                    "birthday": birthday,
                    "company_id": cid,
                    "active": active,
                    "created_at": datetime.now().isoformat()
                }
                if enrollment_image_ids and i < len(enrollment_image_ids):
                    payload["enrollment_image_id"] = enrollment_image_ids[i]
                
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.success(f"Upserted {len(points)} samples for {user_name} (ID: {user_id}, Company: {cid})")
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
            "company_id": None,
            "score": 0.0,
            "detect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vector_count": 0
        }
        
        try:
            vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)

            # Build query filter to only include active users
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="active",
                        match=models.MatchValue(value=True)
                    )
                ]
            )

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
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
            u_id = payload.get("user_id")
            user_id_str = str(u_id) if u_id is not None else "Unknown"

            # Count total vectors for this user
            count_res = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id_str))]
                )
            )

            return {
                "name": payload.get("user_name", "Unknown"),
                "user_id": user_id_str,
                "birthday": payload.get("birthday", "N/A"),
                "company_id": payload.get("company_id", "Unknown"),
                "score": score,
                "detect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vector_count": count_res.count
            }

        except Exception as e:
            logger.error(f"Error during recognition: {e}")
            return default_res

    def get_all_users(self, company_id: Optional[str] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieves unique users. Optional filtering by company_id and active status.
        """
        try:
            all_users = {}
            offset = None
            
            must_conditions = []
            
            if company_id and company_id != "ALL":
                if isinstance(company_id, list):
                    must_conditions.append(models.FieldCondition(key="company_id", match=models.MatchAny(any=[str(c) for c in company_id])))
                else:
                    must_conditions.append(models.FieldCondition(key="company_id", match=models.MatchValue(value=str(company_id))))
                    
            if active_only:
                # If active_only is True, we only fetch points where active is not False
                # Because old data might not have 'active' field, we could just filter where active == True
                must_conditions.append(models.FieldCondition(key="active", match=models.MatchValue(value=True)))

            scroll_filter = models.Filter(must=must_conditions) if must_conditions else None
            
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
                    if u_id is not None:
                        u_id_str = str(u_id)
                        if u_id_str not in all_users:
                            all_users[u_id_str] = {
                                "user_id": u_id_str,
                                "user_name": payload.get("user_name"),
                                "birthday": payload.get("birthday"),
                                "active": payload.get("active", True) # Default to True if missing
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
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
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

    def set_user_active_status(self, user_id: str, active: bool) -> bool:
        """
        Update the 'active' status for all points of a given user.
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"active": active},
                points=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
                )
            )
            logger.success(f"Set active={active} for user ID: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set active status for user {user_id}: {e}")
            return False

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
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
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
                        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
                    )
                )
            )
            logger.success(f"Deleted user ID: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    def get_user_points(self, user_id: str) -> List[models.Record]:
        """
        Retrieve all points (embeddings) for a specific user_id.
        """
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))]
                ),
                limit=100, # Adjust limit as needed, or implement full pagination
                with_payload=True,
                with_vectors=False # No need for vectors in this case
            )
            return records
        except Exception as e:
            logger.error(f"Failed to get points for user {user_id}: {e}")
            return []

    def delete_point(self, point_id: str) -> bool:
        """
        Delete a specific point (embedding) by its ID.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsSelector(
                    points=[point_id]
                )
            )
            logger.success(f"Deleted point ID: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete point {point_id}: {e}")
            return False

    def delete_user_points(self, user_id: str) -> bool:
        """
        Deletes all points (embeddings) associated with a specific user_id.
        """
        try:
            response = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=str(user_id)),
                            )
                        ]
                    )
                ),
            )
            if response.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Successfully deleted all Qdrant points for user_id: {user_id}")
                return True
            else:
                logger.warning(f"Failed to delete Qdrant points for user_id {user_id}: {response.status}")
                return False
        except Exception as e:
            logger.error(f"Error deleting all Qdrant points for user_id {user_id}: {e}")
            return False

# Global instance for shared use
attendance = QdrantAttendanceManager()

__all__ = ["QdrantAttendanceManager", "attendance"]
