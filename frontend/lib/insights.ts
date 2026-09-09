import { AnalyzeResponse } from "./types";

export function lightingInsight(lighting: string): string {
  if (lighting.includes("Natural") || lighting.includes("Excellent")) {
    return "Natural Light Advantage: Position furniture to take advantage of natural light. Add sheer curtains to control glare and blackout curtains for privacy and sleep.";
  }
  if (lighting.includes("Good")) {
    return "Lighting Balance: Mix ambient, task, and accent lighting. Use warm white (2700-3000K) for living areas and cool white (4000-5000K) for workspaces.";
  }
  if (lighting.includes("Moderate")) {
    return "Add Layers: Your room relies on artificial light. Layer floor lamps and table lamps with your overhead fixture to avoid flat, harsh lighting.";
  }
  return "Brighten Up: Add multiple light sources! Use overhead lighting, floor lamps, and table lamps. Aim for 200-300 lumens per square meter in living spaces.";
}

export function densityInsight(objectDensity: number, detectionCount: number): string {
  if (detectionCount === 0) {
    return "Open Canvas: We couldn't confidently detect existing furniture in this shot — great news if you're starting fresh, or try a wider angle for a fuller read on the space.";
  }
  if (objectDensity < 0.08) {
    return "Open & Spacious: Detected furniture covers a small share of the frame. You have room to add pieces without crowding the space.";
  }
  if (objectDensity < 0.2) {
    return "Balanced Layout: Furniture coverage looks moderate. Keep sightlines and walking paths clear as you add or rearrange pieces.";
  }
  return "Consider Decluttering: Detected furniture covers a large share of the frame. Removing or consolidating a piece or two can make the room feel more open.";
}

export function varietyInsight(analysis: AnalyzeResponse): string | null {
  const labels = new Set(analysis.detections.map((d) => d.label));
  if (labels.size >= 5) {
    return `Rich Detail: We picked up ${labels.size} distinct object types (${Array.from(labels).slice(0, 5).join(", ")}${labels.size > 5 ? ", ..." : ""}) — plenty of existing character to design around.`;
  }
  return null;
}

export function buildInsights(analysis: AnalyzeResponse): string[] {
  const insights = [
    lightingInsight(analysis.lighting),
    densityInsight(analysis.objectDensity, analysis.detections.length),
  ];
  const variety = varietyInsight(analysis);
  if (variety) insights.push(variety);
  return insights;
}
