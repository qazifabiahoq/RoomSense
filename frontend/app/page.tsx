"use client";

import { useState } from "react";
import Header from "@/components/Header";
import RoomPicker from "@/components/RoomPicker";
import UploadZone from "@/components/UploadZone";
import DetectionOverlayImage from "@/components/DetectionOverlayImage";
import MetricsRow from "@/components/MetricsRow";
import DetectedObjects from "@/components/DetectedObjects";
import ColorPalette from "@/components/ColorPalette";
import InsightsList from "@/components/InsightsList";
import Recommendations from "@/components/Recommendations";
import PaletteSuggestions from "@/components/PaletteSuggestions";
import RedesignSection from "@/components/RedesignSection";
import ShareBar from "@/components/ShareBar";
import { buildInsights } from "@/lib/insights";
import { API_BASE_URL } from "@/lib/config";
import { AnalyzeResponse, RoomType } from "@/lib/types";

type Mode = "upload" | "manual";

export default function Home() {
  const [roomType, setRoomType] = useState<RoomType>("Living Room");
  const [mode, setMode] = useState<Mode>("upload");

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setAnalysis(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(90000),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Analysis failed (${res.status})`);
      }

      const data: AnalyzeResponse = await res.json();
      setAnalysis(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.name === "TimeoutError" || err.name === "AbortError"
            ? "The analysis engine is waking up (free hosting sleeps when idle). Please try again in a moment."
            : err.message
          : "Something went wrong analyzing your photo.";
      setError(message);
    } finally {
      setAnalyzing(false);
    }
  }

  const insights = analysis ? buildInsights(analysis) : [];

  return (
    <main className="min-h-screen pb-24">
      <Header />

      <div className="mx-auto mt-8 max-w-5xl space-y-8 px-4 sm:px-6">
        <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6 sm:p-8">
          <div className="grid gap-8 sm:grid-cols-2">
            <RoomPicker value={roomType} onChange={setRoomType} />
            <div>
              <p className="mb-3 text-sm font-semibold text-neutral-900">How do you want to start?</p>
              <div className="flex gap-2">
                {(["upload", "manual"] as Mode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex-1 rounded-xl border-2 px-4 py-2.5 text-sm font-semibold transition-colors ${
                      mode === m
                        ? "border-neutral-900 bg-neutral-900 text-white"
                        : "border-neutral-200 bg-white text-neutral-700 hover:border-neutral-400"
                    }`}
                  >
                    {m === "upload" ? "Upload a Photo" : "Skip — Just Recommendations"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {mode === "upload" && (
          <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6 sm:p-8">
            <h2 className="font-display mb-1 text-xl font-bold text-neutral-900">Upload a Photo of Your Room</h2>
            <p className="mb-5 text-sm text-neutral-600">
              Our vision model detects real furniture and fixtures, measures lighting, and extracts your room&apos;s
              actual color palette — no guessing.
            </p>

            {!previewUrl && <UploadZone onFileSelected={handleFile} disabled={analyzing} />}

            {previewUrl && (
              <div className="space-y-4">
                {analysis ? (
                  <DetectionOverlayImage src={previewUrl} detections={analysis.detections} alt="Analyzed room" />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt="Uploaded room" className="w-full rounded-2xl border-2 border-neutral-200" />
                )}

                {analyzing && (
                  <div className="flex items-center justify-center gap-2 rounded-xl bg-blue-50 py-3 text-sm font-semibold text-blue-700">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-300 border-t-blue-700" />
                    Running real object detection & color analysis…
                  </div>
                )}

                {error && (
                  <div className="rounded-xl bg-red-50 px-4 py-3 text-sm font-medium text-red-700">{error}</div>
                )}

                <button
                  onClick={() => {
                    setPreviewUrl(null);
                    setAnalysis(null);
                    setError(null);
                  }}
                  className="text-sm font-semibold text-neutral-500 underline hover:text-neutral-900"
                >
                  Choose a different photo
                </button>
              </div>
            )}
          </div>
        )}

        {analysis && mode === "upload" && (
          <>
            <MetricsRow
              items={[
                { icon: "🎯", label: "Design Focus", value: roomType },
                { icon: "🪑", label: "Objects Detected", value: String(analysis.detections.length) },
                { icon: "💡", label: "Lighting", value: analysis.lighting },
                { icon: "✓", label: "Detection Confidence", value: `${Math.round(analysis.avgConfidence * 100)}%` },
              ]}
            />

            <div className="grid gap-6 sm:grid-cols-2">
              <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6">
                <h3 className="font-display mb-3 text-lg font-bold text-neutral-900">Detected Furniture & Fixtures</h3>
                <DetectedObjects detections={analysis.detections} />
              </div>
              <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6">
                <ColorPalette title="Your Room's Actual Colors" colors={analysis.colorPalette} />
              </div>
            </div>

            <InsightsList insights={insights} />
          </>
        )}

        {(mode === "manual" || (mode === "upload" && analysis)) && (
          <>
            <Recommendations roomType={roomType} />
            <PaletteSuggestions roomType={roomType} />
          </>
        )}

        {analysis && previewUrl && mode === "upload" && (
          <RedesignSection roomType={roomType} originalImageUrl={previewUrl} />
        )}

        {(mode === "manual" || analysis) && (
          <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6 text-center sm:p-8">
            <p className="mb-4 text-sm font-semibold text-neutral-600">Share Your Design</p>
            <ShareBar roomType={roomType} />
          </div>
        )}
      </div>
    </main>
  );
}
