import base64
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from . import recommendations, redesign, vision

_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load real models once at startup (also triggers the one-time weight
    # download) so the first user request isn't the one paying that cost.
    _models["scene_classifier"] = vision.SceneClassifier()
    _models["object_detector"] = vision.ObjectDetector()
    yield
    _models.clear()


app = FastAPI(title="RoomSense API", lifespan=lifespan)

allowed_origins = os.environ.get("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOM_TYPES = list(recommendations.ROOM_CONFIGS.keys())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/room-types")
def get_room_types():
    return {"room_types": ROOM_TYPES}


@app.get("/api/styles")
def get_styles():
    return {
        "styles": [
            {"name": name, "description": cfg["description"], "colors": cfg["colors"]}
            for name, cfg in redesign.STYLES.items()
        ]
    }


@app.post("/api/analyze")
async def analyze_room(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    scene_classifier = _models["scene_classifier"]
    object_detector = _models["object_detector"]

    scene = scene_classifier.classify(image)
    detected_objects, detection_boxes = object_detector.detect(image)
    lighting = vision.analyze_lighting(image)
    dimensions = vision.estimate_dimensions(image, scene["room_type"], detection_boxes)
    color_palette = vision.extract_color_palette(image)

    width, length = dimensions["width"], dimensions["length"]
    aspect = width / length if length else 1.0
    if 0.9 <= aspect <= 1.1:
        layout_type = "Square"
    elif aspect > 1.5 or aspect < 0.67:
        layout_type = "Elongated"
    else:
        layout_type = "Rectangular"

    return {
        "room_type": scene["room_type"],
        "confidence": round(scene["confidence"], 4),
        "dimensions": dimensions,
        "lighting": lighting,
        "layout_type": layout_type,
        "detected_objects": detected_objects,
        "color_palette": color_palette,
        "scene_top_predictions": scene["top_predictions"][:5],
    }


class RecommendationsRequest(BaseModel):
    room_type: str
    area: float
    lighting: str
    height: float = 2.6


@app.post("/api/recommendations")
def get_recommendations(req: RecommendationsRequest):
    return {
        "recommendations": recommendations.generate_room_recommendations(req.room_type),
        "insights": recommendations.generate_detailed_insights(req.room_type, req.area, req.lighting, req.height),
        "color_palette_suggestions": recommendations.get_palette_suggestions(req.room_type),
    }


class RedesignRequestBody(BaseModel):
    style: str
    room_type: str


@app.post("/api/redesign")
def generate_redesign(req: RedesignRequestBody):
    if req.style not in redesign.STYLES:
        raise HTTPException(status_code=400, detail="Unknown style.")
    try:
        image = redesign.redesign_room(req.style, req.room_type)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "style": req.style,
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
    }
