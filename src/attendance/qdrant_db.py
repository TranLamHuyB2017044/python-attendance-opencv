from typing import List, Optional, Tuple, Dict
import numpy as np
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
            location=QdrantConfig.LOCATION
        )
        self.collection_name = QdrantConfig.COLLECTION_NAME
        self.threshold = RecognitionConfig.THRESHOLD
        self.vector_size = RecognitionConfig.EMBEDDING_DIM
        
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
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
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            raise e

    def upsert_user(self, user_name: str, user_id: str, embeddings: List[np.ndarray], user_info: Optional[Dict] = None) -> bool:
        """
        Save/Update user embeddings in Qdrant.
        """
        try:
            points = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    vector = emb.tolist()
                else:
                    vector = list(emb)

                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "user_name": user_name,
                        "user_info": user_info or {}
                    }
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.success(f"Successfully upserted {len(points)} embeddings for user '{user_name}' (ID: {user_id}) to Qdrant.")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert user '{user_name}' to Qdrant: {e}")
            return False

    def recognize(self, query_embedding: np.ndarray) -> Tuple[str, str, float]:
        """
        Search for the closest face in Qdrant.
        Returns: (user_name, user_id, score)
        """
        try:
            if isinstance(query_embedding, np.ndarray):
                vector = query_embedding.tolist()
            else:
                vector = list(query_embedding)

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=1,
                with_payload=True
            )

            if not search_result:
                return "Unknown", "Unknown", 0.0

            result = search_result[0]
            score = result.score
            user_name = result.payload.get("user_name", "Unknown")
            user_id = result.payload.get("user_id", "Unknown")

            if score >= self.threshold:
                return user_name, user_id, score
            
            return "Unknown", "Unknown", score

        except Exception as e:
            logger.error(f"Error during Qdrant search: {e}")
            return "Unknown", 0.0

__all__ = ["QdrantAttendanceManager"]
