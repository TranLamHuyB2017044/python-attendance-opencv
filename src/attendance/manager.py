"""
Attendance Manager module.
Handles user enrollment, local embedding storage, and face matching.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EMBEDDINGS_DIR, RecognitionConfig


class AttendanceManager:
    """
    Manager for user face embeddings and identification.
    
    This class handles:
    - Loading/Saving known faces from disk
    - Registering new users (Enrollment)
    - Matching faces using Cosine Similarity
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Attendance Manager.
        
        Args:
            db_path: Path to the .pkl file storing embeddings
        """
        self.db_path = db_path or str(EMBEDDINGS_DIR / "users.pkl")
        self.known_embeddings: Dict[str, np.ndarray] = {}  # {user_id: embedding}
        self.threshold = RecognitionConfig.THRESHOLD
        
        self.load_database()

    def load_database(self) -> None:
        """Load embeddings from pickle file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.known_embeddings)} users from database.")
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                self.known_embeddings = {}
        else:
            logger.info("No existing database found. Starting fresh.")

    def save_database(self) -> None:
        """Save embeddings to pickle file."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            logger.info(f"Database saved with {len(self.known_embeddings)} users.")
        except Exception as e:
            logger.error(f"Error saving database: {e}")

    def enroll_user(self, user_name: str, embedding: np.ndarray) -> bool:
        """
        Register a new user with their face embedding.
        
        Args:
            user_name: Unique name or ID of the user
            embedding: 512-D vector from InsightFace
            
        Returns:
            True if successful
        """
        if embedding.shape[0] != RecognitionConfig.EMBEDDING_DIM:
            logger.error(f"Invalid embedding dimension: {embedding.shape[0]}")
            return False
            
        self.known_embeddings[user_name] = embedding
        self.save_database()
        logger.success(f"User '{user_name}' enrolled successfully.")
        return True

    def recognize(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Match a query embedding against known faces.
        
        Args:
            query_embedding: Embedding to check
            
        Returns:
            Tuple of (Name, Similarity Score)
            If no match found, returns ("Unknown", 0.0)
        """
        if not self.known_embeddings:
            return "Unknown", 0.0

        names = list(self.known_embeddings.keys())
        embeddings = np.array(list(self.known_embeddings.values()))
        
        # Reshape query for scikit-learn
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.threshold:
            return names[best_idx], best_score
            
        return "Unknown", best_score


__all__ = ["AttendanceManager"]
