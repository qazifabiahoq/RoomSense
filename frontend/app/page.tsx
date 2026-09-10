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
import InspirationGallery from "@/components/InspirationGallery";
import ShareBar from "@/components/ShareBar";
import { buildInsights } from "@/lib/insights";
import { API_BASE_URL } from "@/lib/config";
import { AnalyzeResponse, RoomType } from "@/lib/types";

export default function Home() {
  const [roomType, setRoomType] = useState<RoomType>("Living Room");
  const [skipped, setSkipped] = useState(false);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setAnalysis(null);
    setSkipped(false);
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
            ? "The analysis service is waking up. This can take up to a minute on the free tier, please try again shortly."
            : err.message
          : "Something went wrong analyzing your photo.";
      setError(message);
    } finally {
      setAnalyzing(false);
    }
  }

  const insights = analysis ? buildInsights(analysis) : [];
  const showRecommendations = skipped || Boolean(analysis);

  return (
    <main className="min-h-screen pb-20">
      <Header />

      <div className="mx-auto mt-6 max-w-2xl space-y-6 px-4 sm:mt-10 sm:px-6">
        <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-6">
          <RoomPicker
            value={roomType}
            onChange={(value) => {
              setRoomType(value);
            }}
          />

          {!previewUrl && (
            <div className="mt-5">
              <UploadZone onFileSelected={handleFile} disabled={analyzing} />
              {!skipped && (
                <button
                  onClick={() => setSkipped(true)}
                  className="mx-auto mt-3 block text-sm text-neutral-400 underline decoration-neutral-300 underline-offset-2 hover:text-neutral-700"
                >
                  Continue without a photo
                </button>
              )}
            </div>
          )}

          {previewUrl && (
            <div className="mt-5 space-y-4">
              {analysis ? (
                <DetectionOverlayImage src={previewUrl} detections={analysis.detections} alt="Your room" />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl} alt="Your room" className="w-full rounded-xl border border-neutral-200" />
              )}

              {analyzing && (
                <div className="flex items-center justify-center gap-2 rounded-lg bg-neutral-50 py-3 text-sm font-medium text-neutral-600">
                  <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-neutral-300 border-t-brand-500" />
                  Analyzing your room
                </div>
              )}

              {error && (
                <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
              )}

              <button
                onClick={() => {
                  setPreviewUrl(null);
                  setAnalysis(null);
                  setError(null);
                }}
                className="text-sm text-neutral-400 underline decoration-neutral-300 underline-offset-2 hover:text-neutral-700"
              >
                Choose a different photo
              </button>
            </div>
          )}
        </div>

        {analysis && (
          <>
            <MetricsRow
              items={[
                { icon: "🛋️", label: "Room type", value: roomType },
                { icon: "🪑", label: "Items found", value: String(analysis.detections.length) },
                { icon: "💡", label: "Lighting", value: analysis.lighting },
                { icon: "✓", label: "Confidence", value: `${Math.round(analysis.avgConfidence * 100)}%` },
              ]}
            />

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm">
                <h3 className="font-display mb-3 text-base font-semibold text-neutral-900">What we found</h3>
                <DetectedObjects detections={analysis.detections} />
              </div>
              <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm">
                <ColorPalette title="Your color palette" colors={analysis.colorPalette} />
              </div>
            </div>

            <InsightsList insights={insights} />
          </>
        )}

        {showRecommendations && (
          <>
            <Recommendations roomType={roomType} />
            <PaletteSuggestions roomType={roomType} />
            <InspirationGallery roomType={roomType} />
          </>
        )}

        {analysis && previewUrl && <RedesignSection roomType={roomType} originalImageUrl={previewUrl} />}

        {showRecommendations && (
          <div className="rounded-xl border border-neutral-200 bg-white p-5 text-center shadow-sm sm:p-6">
            <p className="mb-3 text-sm font-medium text-neutral-500">Share this</p>
            <ShareBar roomType={roomType} />
          </div>
        )}
      </div>
    </main>
  );
}
