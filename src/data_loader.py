import os
import zipfile
import requests
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def download_coco_dataset():
    """Download COCO dataset with better error handling"""
    zip_path = "coco-2017-dataset.zip"
    extract_path = "."

    if not os.path.exists(zip_path):
        print("Downloading COCO dataset...")
        try:
            result = subprocess.run([
                'kaggle', 'datasets', 'download', '-d', 'awsaf49/coco-2017-dataset'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print("Kaggle download failed, trying alternative approach...")
                return download_demo_images()

        except Exception as e:
            print(f"Download failed: {e}")
            return download_demo_images()

    if os.path.exists(zip_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return True
    return False

def download_demo_images():
    """Download a few sample images for demo purposes"""
    print("Downloading demo images...")
    demo_urls = [
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        # ... your other URLs
    ]

    os.makedirs("demo_images", exist_ok=True)
    downloaded_paths = []

    for i, url in enumerate(tqdm(demo_urls, desc="Downloading demo images")):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                img_path = f"demo_images/image_{i:03d}.jpg"
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                downloaded_paths.append(img_path)
        except Exception as e:
            print(f"Failed to download image {i}: {e}")

    return downloaded_paths

def load_images(max_images=1000):
    """Load images with fallback options"""
    print("Loading images...")
    image_paths = []
    
    possible_paths = [
        "val2017/*.jpg",
        "train2017/*.jpg", 
        "demo_images/*.jpg",
        "*.jpg",
        "images/*.jpg"
    ]

    for path_pattern in possible_paths:
        paths = list(Path('.').glob(path_pattern))
        if paths:
            image_paths.extend(paths)
            print(f"Found {len(paths)} images in {path_pattern}")

    if not image_paths:
        print("No images found in dataset, downloading demo images...")
        image_paths = download_demo_images()

    image_paths = image_paths[:max_images]
    
    images = []
    valid_paths = []
    for i, path in enumerate(tqdm(image_paths, desc="Loading images")):
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if len(images) == 0:
        raise Exception("No valid images could be loaded!")

    print(f"Successfully loaded {len(images)} images")
    return images, valid_paths
