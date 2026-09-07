"use client";

import { useEffect, useRef, useState } from "react";
import {
  AnalysisResult,
  RecommendationsResult,
  analyzeRoomPhoto,
  getRecommendations,
  getRoomTypes,
} from "@/lib/api";
import AnalysisResults from "@/components/AnalysisResults";
import RedesignPanel from "@/components/RedesignPanel";

type Mode = "upload" | "manual";

export default function Home() {
  const [roomTypes, setRoomTypes] = useState<string[]>([]);
  const [targetRoomType, setTargetRoomType] = useState("Living Room");
  const [mode, setMode] = useState<Mode>("upload");

  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [recs, setRecs] = useState<RecommendationsResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [manualWidth, setManualWidth] = useState(4.0);
  const [manualLength, setManualLength] = useState(4.5);
  const [manualHeight, setManualHeight] = useState(2.6);
  const [manualLighting, setManualLighting] = useState("Good");

  useEffect(() => {
    getRoomTypes()
      .then((res) => {
        setRoomTypes(res.room_types);
        if (res.room_types.length) setTargetRoomType(res.room_types[0]);
      })
      .catch(() => setRoomTypes([]));
  }, []);

  async function handleFile(file: File) {
    setError(null);
    setAnalysis(null);
    setRecs(null);
    setImagePreview(URL.createObjectURL(file));
    setLoading(true);
    try {
      const result = await analyzeRoomPhoto(file);
      setAnalysis(result);
      const recsResult = await getRecommendations(
        targetRoomType,
        result.dimensions.area,
        result.lighting,
        result.dimensions.height
      );
      setRecs(recsResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  async function handleManualSubmit() {
    setError(null);
    setLoading(true);
    try {
      const area = Math.round(manualWidth * manualLength * 10) / 10;
      const manualAnalysis: AnalysisResult = {
        room_type: targetRoomType,
        confidence: 1.0,
        dimensions: {
          width: manualWidth,
          length: manualLength,
          height: manualHeight,
          area,
          method: "manual_entry",
          note: "You entered these measurements directly.",
        },
        lighting: `${manualLighting} lighting`,
        layout_type: "User Specified",
        detected_objects: [],
        color_palette: [],
        scene_top_predictions: [],
      };
      setAnalysis(manualAnalysis);
      const recsResult = await getRecommendations(targetRoomType, area, manualAnalysis.lighting, manualHeight);
      setRecs(recsResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate recommendations.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <div className="header">
        <h1 className="logo">RoomSense</h1>
        <p className="tagline">Design your perfect space</p>
        <span className="badge">Real AI Room Analysis</span>
      </div>

      <div className="hero">
        <img
          src="https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?w=1200&h=400&fit=crop&q=80"
          alt="Modern living room interior"
        />
      </div>

      <div className="grid-2">
        <div className="field">
          <label>What room are you designing?</label>
          <select value={targetRoomType} onChange={(e) => setTargetRoomType(e.target.value)}>
            {roomTypes.map((rt) => (
              <option key={rt} value={rt}>{rt}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>How do you want to analyze?</label>
          <div className="tabs">
            <button className={`tab ${mode === "upload" ? "active" : ""}`} onClick={() => setMode("upload")}>
              Upload / Take Photo
            </button>
            <button className={`tab ${mode === "manual" ? "active" : ""}`} onClick={() => setMode("manual")}>
              Manual Entry
            </button>
          </div>
        </div>
      </div>

      {mode === "upload" && (
        <div className="card">
          <h3>Upload a Photo of Your Room</h3>
          <p className="note" style={{ marginBottom: "1rem" }}>
            On a phone, your browser's file picker will offer a camera option too.
          </p>
          <div className="dropzone" onClick={() => fileInputRef.current?.click()}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
            <p style={{ margin: 0, fontWeight: 600 }}>Click to choose a photo</p>
          </div>

          {imagePreview && (
            <div className="grid-2" style={{ marginTop: "1.5rem" }}>
              <div>
                <img className="uploaded-image" src={imagePreview} alt="Uploaded room" />
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                {loading && <p className="spinner-text">Analyzing your space with real computer-vision models...</p>}
                {!loading && analysis && <span className="status">✓ Analysis Complete</span>}
              </div>
            </div>
          )}
        </div>
      )}

      {mode === "manual" && (
        <div className="card">
          <h3>Enter Your Room Dimensions</h3>
          <p className="note" style={{ marginBottom: "1rem" }}>No photo? Just tell us about your room.</p>
          <div className="grid-2">
            <div className="field">
              <label>Width (meters)</label>
              <input type="number" min={2} max={15} step={0.1} value={manualWidth}
                onChange={(e) => setManualWidth(parseFloat(e.target.value))} />
            </div>
            <div className="field">
              <label>Length (meters)</label>
              <input type="number" min={2} max={15} step={0.1} value={manualLength}
                onChange={(e) => setManualLength(parseFloat(e.target.value))} />
            </div>
            <div className="field">
              <label>Height (meters)</label>
              <input type="number" min={2} max={5} step={0.1} value={manualHeight}
                onChange={(e) => setManualHeight(parseFloat(e.target.value))} />
            </div>
            <div className="field">
              <label>How's the lighting?</label>
              <select value={manualLighting} onChange={(e) => setManualLighting(e.target.value)}>
                <option>Poor</option>
                <option>Moderate</option>
                <option>Good</option>
                <option>Excellent</option>
              </select>
            </div>
          </div>
          <button className="btn" style={{ marginTop: "1.5rem" }} onClick={handleManualSubmit} disabled={loading}>
            {loading ? "Generating..." : "Generate Design Recommendations"}
          </button>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {analysis && recs && (
        <>
          <AnalysisResults
            analysis={analysis}
            recommendations={recs.recommendations}
            insights={recs.insights}
            paletteSuggestions={recs.color_palette_suggestions}
          />
          <RedesignPanel roomType={targetRoomType} />
        </>
      )}

      <div className="disclosure">
        <strong>How the analysis works:</strong> Room type and confidence come from a real Places365
        scene-classification model. Detected objects come from a real COCO-trained object detector.
        Lighting is measured directly from your photo's brightness and color temperature. Room
        dimensions are estimated using a detected object of known size as a reference (or a typical
        average size for the room type when no reference object is found) — a single 2D photo can't
        give exact measurements, so treat these as estimates and use Manual Entry for anything precise.
        The color palette is extracted from your photo's actual pixels. Nothing here is randomly
        generated.
      </div>
    </main>
  );
}
