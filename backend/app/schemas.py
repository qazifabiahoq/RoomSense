from typing import List, Dict, Optional

from pydantic import BaseModel


class Dimensions(BaseModel):
    width: float
    length: float
    height: float
    area: float
    method: str
    note: str


class RoomAnalysisResponse(BaseModel):
    room_type: str
    confidence: float
    dimensions: Dimensions
    lighting: str
    layout_type: str
    detected_objects: List[str]
    color_palette: List[str]
    scene_top_predictions: List[Dict[str, float]]


class RecommendationResponse(BaseModel):
    zone_name: str
    location: str
    furniture: List[str]
    lighting_needs: str
    considerations: List[str]


class InsightsAndRecommendationsResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    insights: List[str]
    color_palette_suggestions: List[Dict[str, object]]


class RedesignRequest(BaseModel):
    style: str
    room_type: str
    image_base64: str


class RedesignResponse(BaseModel):
    image_base64: str
    style: str
