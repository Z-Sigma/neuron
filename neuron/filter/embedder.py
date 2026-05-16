from typing import List, Optional
import logging
import openai
import threading
from neuron.config import settings

logger = logging.getLogger(__name__)
_model_lock = threading.Lock()

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
            with _model_lock:
                try:
                    self.local_model = SentenceTransformer(self.model_name)
                    self.mode = "local"
                    logger.info(f"Using local embedder: {self.model_name}")
                except Exception as e:
                    logger.error(f"Local model load failed: {e}")
                    self.mode = "mock"
                    self.dimension = settings.embedding_dimension
                    if "text-embedding-3-small" in settings.embedding_model_name:
                        self.dimension = 1536
                    elif "text-embedding-3-large" in settings.embedding_model_name:
                        self.dimension = 3072
        elif api_key and api_key != "dummy":
            self.client = openai.OpenAI(api_key=api_key)
            self.model = settings.embedding_model_name if "text-embedding" in settings.embedding_model_name else "text-embedding-3-small"
            self.mode = "openai"
        else:
            self.mode = "mock"
            self.dimension = settings.embedding_dimension
            # Detect dimension if a known OpenAI model is used in settings
            if "text-embedding-3-small" in settings.embedding_model_name:
                self.dimension = 1536
            elif "text-embedding-3-large" in settings.embedding_model_name:
                self.dimension = 3072
            logger.warning(f"No API key and local models unavailable. Using Mock Embedder ({self.dimension}d).")

    def embed(self, text: str) -> List[float]:
        if self.mode == "local":
            with _model_lock:
                return self.local_model.encode(text).tolist()
        elif self.mode == "openai":
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        else:
            import hashlib
            h = hashlib.md5(text.encode()).digest()
            vec = []
            for i in range(self.dimension):
                vec.append((h[i % len(h)] / 255.0) - 0.5)
            return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "local":
            with _model_lock:
                return self.local_model.encode(texts).tolist()
        elif self.mode == "openai":
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        else:
            return [self.embed(t) for t in texts]
