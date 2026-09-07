"use client";

import { AnalysisResult, PaletteSuggestion, Recommendation } from "@/lib/api";

function MetricBox({ icon, label, value, unit }: { icon: string; label: string; value: string | number; unit?: string }) {
  return (
    <div className="metric-box">
      <div style={{ fontSize: "1.4rem", marginBottom: "0.4rem" }}>{icon}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
    </div>
  );
}

export default function AnalysisResults({
  analysis,
  recommendations,
  insights,
  paletteSuggestions,
}: {
  analysis: AnalysisResult;
  recommendations: Recommendation[];
  insights: string[];
  paletteSuggestions: PaletteSuggestion[];
}) {
  const { dimensions } = analysis;

  return (
    <div>
      <h2 className="section-title">Your Room Analysis</h2>

      <div className="metrics">
        <MetricBox icon="🏠" label="Detected Room Type" value={analysis.room_type} />
        <MetricBox icon="📏" label="Estimated Area" value={dimensions.area} unit="m²" />
        <MetricBox icon="💡" label="Lighting" value={analysis.lighting} />
        <MetricBox icon="✓" label="Scene Confidence" value={Math.round(analysis.confidence * 100)} unit="%" />
      </div>

      <div className="card">
        <div className="grid-2">
          <div>
            <h3>Room Specifications</h3>
            <ul>
              <li><strong>Layout Shape:</strong> {analysis.layout_type}</li>
              <li><strong>Width:</strong> {dimensions.width}m</li>
              <li><strong>Length:</strong> {dimensions.length}m</li>
              <li><strong>Height:</strong> {dimensions.height}m</li>
              <li><strong>Total Area:</strong> {dimensions.area}m²</li>
            </ul>
            <p className="note">{dimensions.note}</p>

            {analysis.detected_objects.length > 0 && (
              <>
                <h3 style={{ marginTop: "1.25rem" }}>Detected Objects</h3>
                <div>
                  {analysis.detected_objects.map((obj) => (
                    <span key={obj} className="zone-tag">{obj}</span>
                  ))}
                </div>
              </>
            )}
          </div>
          <div>
            <h3>Scene Classification Confidence</h3>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: `${analysis.confidence * 100}%` }} />
            </div>
            <p className="note">
              {Math.round(analysis.confidence * 100)}% confident this is a {analysis.room_type.toLowerCase()}
              , based on a Places365 scene-classification model's real prediction for your photo.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 style={{ marginBottom: "0.5rem" }}>Smart Recommendations</h2>
        <p className="note" style={{ marginBottom: "1rem" }}>
          Based on a {dimensions.area}m² room with {analysis.lighting.toLowerCase()} conditions
        </p>
        {recommendations.map((rec) => (
          <div key={rec.zone_name} className="rec-item">
            <div className="rec-title">{rec.zone_name}</div>
            <p><strong>Optimal Location:</strong> {rec.location}</p>
            <p><strong>Lighting Setup:</strong> {rec.lighting_needs}</p>
            <div className="furniture-list">
              <strong>Recommended Furniture:</strong>
              {rec.furniture.map((item) => (
                <div key={item} className="furniture-item">• {item}</div>
              ))}
            </div>
            <div style={{ marginTop: "1rem" }}>
              <strong>Key Considerations:</strong>
              {rec.considerations.map((item) => (
                <div key={item}>• {item}</div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <h2 style={{ marginBottom: "1rem" }}>Smart Insights for Your Space</h2>
        {insights.map((insight) => (
          <p key={insight} dangerouslySetInnerHTML={{ __html: insight.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
        ))}
      </div>

      {analysis.color_palette.length > 0 && (
        <div className="card">
          <h2 style={{ marginBottom: "0.5rem" }}>Colors Detected In Your Photo</h2>
          <p className="note" style={{ marginBottom: "0.75rem" }}>
            Extracted directly from your image's actual pixels using k-means color clustering.
          </p>
          <div className="palette-row">
            {analysis.color_palette.map((color) => (
              <div key={color} className="swatch" style={{ background: color }}>
                <span>{color}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <h2 style={{ marginBottom: "1rem" }}>Suggested Color Palettes for Your Room</h2>
        {paletteSuggestions.map((p) => (
          <div key={p.name}>
            <strong>{p.name}</strong>
            <div className="palette-row">
              {p.colors.map((color) => (
                <div key={color} className="swatch" style={{ background: color }}>
                  <span>{color}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
