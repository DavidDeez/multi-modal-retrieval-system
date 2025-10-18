import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm

class MultiModalEnsemble:
    def __init__(self, device='cuda'):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Models loaded")

    def encode_images(self, images, batch_size=32):
        if len(images) == 0:
            raise ValueError("No images to encode")

        embeddings = []
        for i in tqdm(range(0, len(images), batch_size), desc="CLIP image encoding"):
            batch = images[i:i+batch_size]
            try:
                processed_batch = []
                for img in batch:
                    if isinstance(img, str):
                        img = Image.open(img)
                    processed_batch.append(img.convert('RGB'))

                inputs = self.clip_processor(images=processed_batch, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

        if len(embeddings) == 0:
            raise ValueError("Failed to encode any images")
        return np.vstack(embeddings)

    def encode_text_ensemble(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            clip_features = self.clip_model.get_text_features(**inputs)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_features = clip_features.cpu().numpy()

        st_features = self.text_model.encode([text], convert_to_numpy=True)
        st_features = st_features / np.linalg.norm(st_features, axis=1, keepdims=True)

        combined = np.concatenate([clip_features * 0.7, st_features * 0.3], axis=1)
        return combined / np.linalg.norm(combined, axis=1, keepdims=True)

    def encode_image_query(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert('RGB')
        inputs = self.clip_processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
