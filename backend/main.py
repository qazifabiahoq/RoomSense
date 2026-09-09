"""RoomSense Vision API.

Real computer-vision analysis for uploaded room photos:
- Object detection (SSDLite MobileNetV3, COCO-pretrained) to find real furniture/fixtures
- Real brightness measurement -> lighting classification
- Real dominant color extraction (K-Means over actual pixels)

No random or fabricated outputs: every field in the response is derived from
the actual uploaded image.
"""
import io
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sklearn.cluster import KMeans
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms import functional as TF

app = FastAPI(title="RoomSense Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
_model = ssdlite320_mobilenet_v3_large(weights=_weights)
_model.eval()
_categories = _weights.meta["categories"]

# Map COCO category names to friendlier furniture/fixture labels.
FURNITURE_LABELS = {
    "chair": "Chair",
    "couch": "Sofa",
    "bed": "Bed",
    "dining table": "Table",
    "tv": "TV",
    "toilet": "Toilet",
    "sink": "Sink",
    "refrigerator": "Refrigerator",
    "microwave": "Microwave",
    "oven": "Oven",
    "toaster": "Toaster",
    "potted plant": "Plant",
    "book": "Books",
    "clock": "Clock",
    "vase": "Vase",
    "laptop": "Laptop",
    "backpack": "Backpack",
    "suitcase": "Suitcase",
    "handbag": "Bag",
    "cell phone": "Phone",
    "remote": "Remote",
    "teddy bear": "Toys",
    "bottle": "Bottle",
    "cup": "Cup",
    "bowl": "Bowl",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "bench": "Bench",
    "umbrella": "Umbrella",
    "hair drier": "Hair Dryer",
    "toothbrush": "Toothbrush",
    "wine glass": "Glassware",
    "scissors": "Scissors",
}

# Categories that are irrelevant to room design and should be filtered out.
IGNORED_CATEGORIES = {"person", "N/A", "__background__"}

DETECTION_THRESHOLD = 0.45
MAX_DETECTIONS = 12


class Detection(BaseModel):
    label: str
    confidence: float
    box: List[float]  # [xmin, ymin, xmax, ymax] normalized to 0-1


class AnalyzeResponse(BaseModel):
    width: int
    height: int
    brightness: float
    lighting: str
    colorPalette: List[str]
    detections: List[Detection]
    avgConfidence: float
    objectDensity: float


def classify_lighting(brightness: float) -> str:
    if brightness < 60:
        return "Low Light"
    if brightness < 110:
        return "Artificial - Moderate"
    if brightness < 170:
        return "Mixed - Good"
    return "Natural - Excellent"


def extract_palette(image: Image.Image, n_colors: int = 5) -> List[str]:
    small = image.resize((100, 100))
    pixels = np.array(small).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    return ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in centers]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Please upload a JPEG, PNG, or WEBP image.")

    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 12MB).")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    width, height = image.size
    brightness = float(np.array(image.convert("L")).mean())
    lighting = classify_lighting(brightness)
    palette = extract_palette(image)

    tensor = TF.to_tensor(image)
    with torch.no_grad():
        prediction = _model([tensor])[0]

    detections: List[Detection] = []
    total_conf = 0.0
    covered_area = 0.0

    for box, label_idx, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        score_f = float(score)
        if score_f < DETECTION_THRESHOLD:
            continue
        category = _categories[int(label_idx)]
        if category in IGNORED_CATEGORIES:
            continue

        display_name = FURNITURE_LABELS.get(category, category.title())
        xmin, ymin, xmax, ymax = [float(v) for v in box]
        norm_box = [xmin / width, ymin / height, xmax / width, ymax / height]

        detections.append(
            Detection(label=display_name, confidence=round(score_f, 3), box=norm_box)
        )
        total_conf += score_f
        covered_area += (xmax - xmin) * (ymax - ymin)

        if len(detections) >= MAX_DETECTIONS:
            break

    avg_conf = round(total_conf / len(detections), 3) if detections else 0.0
    object_density = round(covered_area / (width * height), 4) if detections else 0.0

    return AnalyzeResponse(
        width=width,
        height=height,
        brightness=round(brightness, 1),
        lighting=lighting,
        colorPalette=palette,
        detections=detections,
        avgConfidence=avg_conf,
        objectDensity=object_density,
    )
