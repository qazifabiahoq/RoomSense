export type RoomType =
  | "Living Room"
  | "Bedroom"
  | "Kitchen"
  | "Bathroom"
  | "Dining Room"
  | "Home Office"
  | "Kids Room"
  | "Laundry Room";

export interface Zone {
  name: string;
  location: string;
  furniture: string[];
  lighting: string;
  considerations: string[];
}

export interface StyleConfig {
  description: string;
  colors: string[];
  prompt: string;
  negativePrompt: string;
}

export interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number]; // normalized xmin, ymin, xmax, ymax
}

export interface AnalyzeResponse {
  width: number;
  height: number;
  brightness: number;
  lighting: string;
  colorPalette: string[];
  detections: Detection[];
  avgConfidence: number;
  objectDensity: number;
}
