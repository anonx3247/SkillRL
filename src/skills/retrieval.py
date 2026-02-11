"""Semantic skill retrieval using sentence-transformers and FAISS."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .models import Skill


class SkillRetriever:
    """Retrieves relevant skills using semantic similarity.

    Uses sentence-transformers for encoding and FAISS for efficient search.
    Employs cosine similarity via normalized embeddings and dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize retriever.

        Args:
            model_name: SentenceTransformer model to use for encoding
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.indexed_skills: list[Skill] = []

    def index_skills(self, skills: list[Skill]) -> None:
        """Build search index from skills.

        Args:
            skills: List of skills to index
        """
        if not skills:
            self.index = None
            self.indexed_skills = []
            return

        # Encode skills as "{name}: {principle}. {when_to_apply}"
        skill_texts = [
            f"{skill.name}: {skill.principle}. {skill.when_to_apply}"
            for skill in skills
        ]

        # Encode with normalization for cosine similarity
        embeddings = self.model.encode(
            skill_texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Additional normalization for FAISS
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        # Build FAISS index for dot product (equivalent to cosine with normalized vectors)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.indexed_skills = skills

    def retrieve(self, query: str, top_k: int = 3, current_iteration: int = 0) -> list[Skill]:
        """Retrieve most relevant skills for a query.

        Args:
            query: Query text
            top_k: Number of skills to retrieve
            current_iteration: Current iteration number for usage tracking

        Returns:
            List of most relevant skills (up to top_k)
        """
        if self.index is None or not self.indexed_skills:
            return []

        # Cap top_k at number of available skills
        top_k = min(top_k, len(self.indexed_skills))

        # Encode query with normalization
        query_embedding = self.model.encode(
            [query],
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Additional normalization for FAISS
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve skills and update usage tracking
        results = [self.indexed_skills[i] for i in indices[0]]
        for skill in results:
            skill.usage_count += 1
            skill.last_used_iteration = current_iteration

        return results
