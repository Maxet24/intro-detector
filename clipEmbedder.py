import clip
# to avoid OpenMP crash
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cosine, euclidean

class ClipEmbedder:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def embed_image(self, image_or_path):
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
        else:
            raise TypeError("Не тот формат")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)

        return embedding.cpu().numpy().astype(np.float32).flatten()

    def embed_images(self, image_paths):
        images = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(batch)
        return embeddings.cpu().numpy().astype(np.float32)

    def distance(self, emb1, emb2, metric="cosine"):
        if metric == "cosine":
            return cosine(emb1, emb2)
        elif metric == "euclidean":
            return euclidean(emb1, emb2)
        else:
            raise ValueError(f"Метрика не та: {metric}")
