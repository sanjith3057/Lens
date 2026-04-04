import requests
from typing import Optional, Union, List
import time
import io
import torch
from PIL import Image
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

from src.__init__ import logger, SecureConfig

class CLIPEmbedder:
    """Layer 2: Local Unified Multimodal Embedding (RTX 2050 CUDA Optimized)"""
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed. CLIPEmbedder disabled.")
            self.model = None
            self.device = "cpu"
        else:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading local CLIP model: {model_name} on {self.device}...")
                self.model = SentenceTransformer(model_name, device=self.device)
                # Ensure truncation is active at the model level if supported
                if hasattr(self.model, 'max_seq_length'):
                    self.model.max_seq_length = 77
                logger.debug("CLIP model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                self.model = None
                self.device = "cpu"

    def embed_text(self, text: str) -> List[float]:
        if not self.model:
            return [0.0] * 512
        
        # Absolute Safety: 120 chars is ~30-40 tokens. 
        # Clip limit is 77. This is 100% silent.
        safe_text = text[:120].strip()
        
        try:
            # We use convert_to_tensor=True for GPU speed
            return self.model.encode(
                safe_text, 
                convert_to_tensor=True, 
                show_progress_bar=False, 
                device=self.device
            ).tolist()
        except Exception as e:
            # Final fallback to tiny string
            if "sequence length" in str(e).lower():
                return self.model.encode(safe_text[:50], convert_to_tensor=True, device=self.device).tolist()
            logger.error(f"CLIP Text Embedding Error: {e}")
            return [0.0] * 512

    def embed_image(self, image_bytes: bytes) -> List[float]:
        if not self.model:
            return [0.0] * 512
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.model.encode(image, convert_to_tensor=True, show_progress_bar=False, device=self.device).tolist()
        except Exception as e:
            logger.error(f"CLIP Image Embedding Error: {e}")
            return [0.0] * 512

    @staticmethod
    def similarity(vec1: List[float], vec2: List[float]) -> float:
        if util is None:
             return 0.0
        return float(util.cos_sim(vec1, vec2))
