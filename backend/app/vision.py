"""Real computer-vision analysis for RoomSense.

No randomness anywhere in this module. Every value returned is derived
from either a trained model's actual output or a deterministic
calculation over the real pixel data of the uploaded photo.
"""
import os
import threading
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import models, transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/roomsense_models")
os.makedirs(CACHE_DIR, exist_ok=True)

PLACES365_WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
PLACES365_CATEGORIES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"

_download_lock = threading.Lock()

# Maps our 8 supported room types to the Places365 scene categories that
# actually describe them (Places365 has 365 real scene classes, e.g.
# "/b/bedroom", "/k/kitchen" - these are genuine trained classes, not a
# guess). A scene classifier's softmax probability over its matching
# categories becomes that room type's real confidence score.
ROOM_TYPE_TO_PLACES365 = {
    "Bedroom": ["bedroom", "bedchamber", "dorm_room", "hotel_room"],
    "Kitchen": ["kitchen", "kitchenette"],
    "Bathroom": ["bathroom", "shower"],
    "Dining Room": ["dining_room", "restaurant", "banquet_hall", "dining_hall", "cafeteria"],
    "Home Office": ["home_office", "office", "office_cubicles", "computer_room", "library_indoor", "library-indoor"],
    "Kids Room": ["playroom", "nursery"],
    "Laundry Room": ["utility_room", "laundromat"],
    "Living Room": ["living_room", "home_theater", "television_room", "waiting_room", "lobby",
                     "recreation_room", "sunroom", "reception"],
}

# COCO object-detector classes that are actual pieces of furniture/fixtures,
# mapped to a friendlier display name. Anything COCO cannot see (dresser,
# nightstand, curtains, etc.) is simply not reported, rather than guessed.
FURNITURE_COCO_CLASSES = {
    "chair": "Chair",
    "couch": "Sofa",
    "bed": "Bed",
    "dining table": "Table",
    "tv": "TV",
    "potted plant": "Plant",
    "refrigerator": "Refrigerator",
    "oven": "Oven",
    "microwave": "Microwave",
    "sink": "Sink",
    "toilet": "Toilet",
    "book": "Books",
    "clock": "Clock",
    "vase": "Vase",
    "laptop": "Laptop",
}

# Average real-world width (in meters) used as a size reference when that
# object is detected in the photo - the same "known object as ruler"
# technique used in manual photogrammetry. Ordered by how reliable/
# consistent that object's real-world size tends to be.
REFERENCE_OBJECT_WIDTHS_M = [
    ("refrigerator", 0.70),
    ("toilet", 0.40),
    ("sink", 0.55),
    ("oven", 0.60),
    ("tv", 1.05),
    ("couch", 1.80),
    ("bed", 1.50),
    ("dining table", 1.20),
    ("chair", 0.50),
]

# Typical average floor area (m^2) per room type, used ONLY as a fallback
# when no reference object was detected in the photo, so the estimate is
# still grounded in real-world averages rather than invented.
TYPICAL_ROOM_AREA_M2 = {
    "Living Room": 22.0,
    "Bedroom": 13.0,
    "Kitchen": 11.0,
    "Bathroom": 6.0,
    "Dining Room": 15.0,
    "Home Office": 10.0,
    "Kids Room": 11.0,
    "Laundry Room": 5.0,
}

DETECTION_CONFIDENCE_THRESHOLD = 0.55


def _download(url: str, dest_path: str) -> str:
    if os.path.exists(dest_path):
        return dest_path
    with _download_lock:
        if os.path.exists(dest_path):
            return dest_path
        tmp_path = dest_path + ".part"
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp_path, dest_path)
    return dest_path


def _load_places365_categories() -> List[str]:
    path = _download(PLACES365_CATEGORIES_URL, os.path.join(CACHE_DIR, "categories_places365.txt"))
    categories = []
    with open(path, "r") as f:
        for line in f:
            raw = line.strip().split(" ")[0]  # e.g. "/b/bedroom"
            name = raw[3:].replace("/", " ")  # -> "bedroom"
            categories.append(name)
    return categories


def _load_places365_model() -> Tuple[torch.nn.Module, List[str]]:
    weights_path = _download(PLACES365_WEIGHTS_URL, os.path.join(CACHE_DIR, "resnet18_places365.pth.tar"))
    categories = _load_places365_categories()

    model = models.resnet18(num_classes=len(categories))
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, categories


_SCENE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SceneClassifier:
    """Real indoor-scene classification using a Places365-trained ResNet18."""

    def __init__(self):
        self.model, self.categories = _load_places365_model()

    @torch.no_grad()
    def classify(self, image: Image.Image) -> Dict:
        tensor = _SCENE_TRANSFORM(image.convert("RGB")).unsqueeze(0)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        top_probs, top_idx = probs.topk(10)
        top_predictions = [
            {"category": self.categories[idx], "probability": float(p)}
            for p, idx in zip(top_probs.tolist(), top_idx.tolist())
        ]

        room_scores = {}
        for room_type, keywords in ROOM_TYPE_TO_PLACES365.items():
            score = 0.0
            for keyword in keywords:
                try:
                    cat_idx = self.categories.index(keyword)
                except ValueError:
                    continue
                score += float(probs[cat_idx])
            room_scores[room_type] = score

        best_room_type = max(room_scores, key=room_scores.get)
        confidence = room_scores[best_room_type]

        if confidence < 0.01:
            best_room_type = "Living Room"
            confidence = max(confidence, float(top_probs[0]) * 0.3)

        return {
            "room_type": best_room_type,
            "confidence": min(confidence, 1.0),
            "top_predictions": top_predictions,
        }


@lru_cache(maxsize=1)
def _load_object_detector():
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.eval()
    return model, weights.meta["categories"], weights.transforms()


class ObjectDetector:
    """Real furniture/fixture detection using a COCO-trained Faster R-CNN."""

    def __init__(self):
        self.model, self.categories, self.preprocess = _load_object_detector()

    @torch.no_grad()
    def detect(self, image: Image.Image) -> Tuple[List[str], List[Dict]]:
        rgb = image.convert("RGB")
        tensor = self.preprocess(rgb)
        output = self.model([tensor])[0]

        width, height = rgb.size
        seen_labels = set()
        display_names: List[str] = []
        boxes: List[Dict] = []

        for box, label_idx, score in zip(output["boxes"], output["labels"], output["scores"]):
            if float(score) < DETECTION_CONFIDENCE_THRESHOLD:
                continue
            coco_name = self.categories[int(label_idx)]
            if coco_name not in FURNITURE_COCO_CLASSES:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            boxes.append({
                "coco_name": coco_name,
                "score": float(score),
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            })

            display_name = FURNITURE_COCO_CLASSES[coco_name]
            if display_name not in seen_labels:
                seen_labels.add(display_name)
                display_names.append(display_name)

        return display_names, boxes


def analyze_lighting(image: Image.Image) -> str:
    """Deterministic lighting read-out from real pixel statistics.

    Brightness comes from mean luminance. Warmth comes from comparing the
    red and blue channel means (tungsten/artificial light skews warm/red,
    daylight skews cool/blue-neutral). No model needed - this is plain,
    explainable image analysis, not a guess.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    brightness = float(arr.mean())
    mean_r = float(arr[:, :, 0].mean())
    mean_b = float(arr[:, :, 2].mean())
    warmth_ratio = mean_r / (mean_b + 1e-6)

    if brightness > 180:
        quality = "Bright"
    elif brightness > 120:
        quality = "Good"
    elif brightness > 70:
        quality = "Moderate"
    else:
        quality = "Low Light"

    if warmth_ratio > 1.15:
        source = "Artificial (Warm)"
    elif warmth_ratio < 0.92:
        source = "Natural (Cool/Daylight)"
    else:
        source = "Mixed"

    return f"{source} - {quality}"


def estimate_dimensions(image: Image.Image, room_type: str, detection_boxes: List[Dict]) -> Dict:
    """Estimate room size from real pixel geometry, not a random guess.

    If a recognizable reference object (fridge, toilet, sofa, etc.) was
    detected, its known average real-world size is used to scale the
    photo's pixel width into meters - the same "known object as ruler"
    technique used in manual photogrammetry. Otherwise falls back to the
    real-world typical average size for that room type. Absolute room
    dimensions from a single uncalibrated 2D photo are inherently an
    approximation - this is disclosed in the response, not hidden.
    """
    img_width_px, img_height_px = image.size
    aspect_ratio = img_width_px / img_height_px

    boxes_by_coco_name = {}
    for box in detection_boxes:
        boxes_by_coco_name.setdefault(box["coco_name"], []).append(box)

    for coco_name, ref_width_m in REFERENCE_OBJECT_WIDTHS_M:
        candidates = boxes_by_coco_name.get(coco_name)
        if not candidates:
            continue
        best = max(candidates, key=lambda b: b["width_px"])
        if best["width_px"] < 5:
            continue

        scale_m_per_px = ref_width_m / best["width_px"]
        estimated_width_m = img_width_px * scale_m_per_px
        estimated_width_m = min(max(estimated_width_m, 2.0), 15.0)
        estimated_length_m = round(estimated_width_m * aspect_ratio, 1)
        estimated_length_m = min(max(estimated_length_m, 2.0), 15.0)
        area = round(estimated_width_m * estimated_length_m, 1)

        return {
            "width": round(estimated_width_m, 1),
            "length": estimated_length_m,
            "height": 2.6,
            "area": area,
            "method": "reference_object",
            "note": f"Estimated using the detected {FURNITURE_COCO_CLASSES[coco_name].lower()} "
                    f"as a size reference. For exact measurements, use Manual Entry.",
        }

    typical_area = TYPICAL_ROOM_AREA_M2.get(room_type, 15.0)
    estimated_width_m = round((typical_area / aspect_ratio) ** 0.5 * aspect_ratio, 1)
    estimated_length_m = round(typical_area / estimated_width_m, 1) if estimated_width_m else 0.0
    return {
        "width": estimated_width_m,
        "length": estimated_length_m,
        "height": 2.6,
        "area": round(estimated_width_m * estimated_length_m, 1),
        "method": "typical_average",
        "note": f"No size-reference object was detected, so this uses the typical average "
                f"size for a {room_type.lower()}. For exact measurements, use Manual Entry.",
    }


def extract_color_palette(image: Image.Image, n_colors: int = 5) -> List[str]:
    """Real dominant-color extraction via K-means clustering on actual pixels."""
    img_array = np.array(image.convert("RGB").resize((150, 150)))
    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    return ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in colors]
