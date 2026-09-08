const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "https://roomsense-api.onrender.com";

export type Dimensions = {
  width: number;
  length: number;
  height: number;
  area: number;
  method: string;
  note: string;
};

export type ScenePrediction = {
  category: string;
  probability: number;
};

export type AnalysisResult = {
  room_type: string;
  confidence: number;
  dimensions: Dimensions;
  lighting: string;
  layout_type: string;
  detected_objects: string[];
  color_palette: string[];
  scene_top_predictions: ScenePrediction[];
};

export type Recommendation = {
  zone_name: string;
  location: string;
  furniture: string[];
  lighting_needs: string;
  considerations: string[];
};

export type PaletteSuggestion = {
  name: string;
  colors: string[];
};

export type RecommendationsResult = {
  recommendations: Recommendation[];
  insights: string[];
  color_palette_suggestions: PaletteSuggestion[];
};

export type StyleInfo = {
  name: string;
  description: string;
  colors: string[];
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function getRoomTypes(): Promise<{ room_types: string[] }> {
  return apiFetch("/api/room-types");
}

export function getStyles(): Promise<{ styles: StyleInfo[] }> {
  return apiFetch("/api/styles");
}

export function analyzeRoomPhoto(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch("/api/analyze", { method: "POST", body: formData });
}

export function getRecommendations(
  roomType: string,
  area: number,
  lighting: string,
  height: number
): Promise<RecommendationsResult> {
  return apiFetch("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ room_type: roomType, area, lighting, height }),
  });
}

export function generateRedesign(
  style: string,
  roomType: string
): Promise<{ style: string; image_base64: string }> {
  return apiFetch("/api/redesign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ style, room_type: roomType }),
  });
}
