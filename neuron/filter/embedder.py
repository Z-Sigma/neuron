from typing import List, Optional
import logging
import openai
from neuron.config import settings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class Embedder:
    def __init__(self, api_key: str = settings.openai_api_key):
        self.use_local = not api_key or settings.llm_provider in ["groq", "ollama"]
        
        if self.use_local and HAS_SENTENCE_TRANSFORMERS:
            self.model_name = settings.embedding_model_name
            try:
                self.local_model = SentenceTransformer(self.model_name)
                self.mode = "local"
                logger.info(f"Using local embedder: {self.model_name}")
            except Exception as e:
                logger.error(f"Local model load failed: {e}")
                self.mode = "mock"
                logger.warning("Falling back to Mock Embedder.")
        elif api_key and api_key != "dummy":
            self.client = openai.OpenAI(api_key=api_key)
            self.model = settings.embedding_model_name if "text-embedding" in settings.embedding_model_name else "text-embedding-3-small"
            self.mode = "openai"
            logger.info(f"Using OpenAI embedder: {self.model}")
        else:
            self.mode = "mock"
            logger.warning("No API key and local models unavailable. Using Mock Embedder.")

    def embed(self, text: str) -> List[float]:
        if self.mode == "local":
            return self.local_model.encode(text).tolist()
        elif self.mode == "openai":
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        else:
            # Mock embedding: just a deterministic random vector based on text hash
            import hashlib
            h = hashlib.md5(text.encode()).digest()
            # Create a dynamic-dim vector
            vec = []
            for i in range(settings.embedding_dimension):
                vec.append((h[i % len(h)] / 255.0) - 0.5)
            return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "local":
            return self.local_model.encode(texts).tolist()
        elif self.mode == "openai":
            # OpenAI natively supports batch embedding
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        else:
            return [self.embed(t) for t in texts]
