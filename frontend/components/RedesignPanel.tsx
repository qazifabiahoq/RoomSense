"use client";

import { useEffect, useState } from "react";
import { generateRedesign, getStyles, StyleInfo } from "@/lib/api";

export default function RedesignPanel({ roomType }: { roomType: string }) {
  const [styles, setStyles] = useState<StyleInfo[]>([]);
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStyles().then((res) => setStyles(res.styles)).catch(() => setStyles([]));
  }, []);

  async function handleSelect(style: string) {
    setSelectedStyle(style);
    setImageBase64(null);
    setError(null);
    setLoading(true);
    try {
      const result = await generateRedesign(style, roomType);
      setImageBase64(result.image_base64);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2>AI Style Inspiration</h2>
      <p className="note" style={{ marginBottom: "1rem" }}>
        This generates a brand-new {roomType.toLowerCase()} image in your chosen style using a real
        Stable Diffusion model (Pollinations.ai, free, no key needed). It does not edit your uploaded
        photo directly — it's inspiration for what the style could look like.
      </p>

      <div className="style-grid">
        {styles.map((s) => (
          <button
            key={s.name}
            className={`style-card ${selectedStyle === s.name ? "selected" : ""}`}
            onClick={() => handleSelect(s.name)}
          >
            <div className="style-name">{s.name}</div>
            <div className="style-description">{s.description}</div>
            <div style={{ display: "flex", gap: "0.4rem", justifyContent: "center", marginTop: "0.75rem" }}>
              {s.colors.map((c) => (
                <div key={c} style={{ width: 24, height: 24, background: c, borderRadius: 6, border: "2px solid #e0e0e0" }} />
              ))}
            </div>
          </button>
        ))}
      </div>

      {loading && <p className="spinner-text">Generating your {selectedStyle} redesign... this can take 20-30 seconds.</p>}
      {error && <div className="error-box">{error}</div>}

      {imageBase64 && !loading && (
        <div>
          <img
            className="uploaded-image"
            src={`data:image/png;base64,${imageBase64}`}
            alt={`AI generated ${selectedStyle} redesign`}
          />
          <a
            className="btn secondary"
            style={{ display: "block", textAlign: "center", marginTop: "1rem", textDecoration: "none" }}
            href={`data:image/png;base64,${imageBase64}`}
            download={`roomsense_${(selectedStyle || "style").toLowerCase().replace(/\s+/g, "_")}.png`}
          >
            Download Image
          </a>
        </div>
      )}
    </div>
  );
}
