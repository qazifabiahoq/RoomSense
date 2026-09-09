"use client";

import { useState } from "react";
import { REDESIGN_STYLES } from "@/lib/roomData";
import { RoomType } from "@/lib/types";

type Status = "idle" | "loading" | "ready" | "error";

export default function RedesignSection({
  roomType,
  originalImageUrl,
}: {
  roomType: RoomType;
  originalImageUrl: string;
}) {
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [imgKey, setImgKey] = useState(0);

  const promptFor = (style: string) => `${roomType}, ${REDESIGN_STYLES[style].prompt}`;

  const redesignUrl = selectedStyle
    ? `/api/redesign?prompt=${encodeURIComponent(promptFor(selectedStyle))}`
    : null;

  const downloadUrl = selectedStyle
    ? `/api/redesign?prompt=${encodeURIComponent(promptFor(selectedStyle))}&download=1&filename=roomsense-${selectedStyle
        .toLowerCase()
        .replace(/\s+/g, "-")}.png`
    : null;

  function selectStyle(style: string) {
    setSelectedStyle(style);
    setStatus("loading");
    setImgKey((k) => k + 1);
  }

  return (
    <div className="rounded-[24px] border-[3px] border-neutral-900 bg-gradient-to-br from-neutral-50 to-white p-6 sm:p-10">
      <h2 className="font-display mb-1 text-center text-2xl font-bold text-neutral-900 sm:text-3xl">
        AI Room Redesign
      </h2>
      <p className="mb-8 text-center text-neutral-600">
        Real generative AI (Stable Diffusion via Pollinations.ai) reimagines your space — free, no API key.
      </p>

      <p className="mb-3 text-sm font-semibold text-neutral-900">Choose a style</p>
      <div className="grid gap-4 sm:grid-cols-2">
        {Object.entries(REDESIGN_STYLES).map(([name, style]) => (
          <button
            key={name}
            onClick={() => selectStyle(name)}
            className={`rounded-2xl border-[3px] p-5 text-left transition-all hover:-translate-y-1 ${
              selectedStyle === name ? "border-neutral-900 bg-neutral-100 shadow-lg" : "border-neutral-200 bg-white"
            }`}
          >
            <p className="font-display text-lg font-bold text-neutral-900">{name}</p>
            <p className="mb-3 text-sm text-neutral-600">{style.description}</p>
            <div className="flex gap-1.5">
              {style.colors.map((c) => (
                <span key={c} className="h-6 w-6 rounded-md border border-neutral-200" style={{ backgroundColor: c }} />
              ))}
            </div>
          </button>
        ))}
      </div>

      {selectedStyle && redesignUrl && (
        <div className="mt-8">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <p className="mb-2 text-center text-sm font-semibold text-neutral-900">Your Original Room</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={originalImageUrl} alt="Original room" className="w-full rounded-xl border-2 border-neutral-200" />
            </div>
            <div>
              <p className="mb-2 text-center text-sm font-semibold text-neutral-900">
                AI-Generated {selectedStyle}
              </p>
              <div className="relative aspect-square w-full overflow-hidden rounded-xl border-2 border-neutral-200 bg-neutral-100">
                {status === "loading" && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-neutral-500">
                    <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-900" />
                    <p className="animate-shimmer text-xs font-medium">Generating with AI… ~20-30s</p>
                  </div>
                )}
                {status === "error" && (
                  <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-red-600">
                    Generation failed. Try again or pick another style.
                  </div>
                )}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  key={imgKey}
                  src={redesignUrl}
                  alt={`AI redesigned ${roomType} in ${selectedStyle} style`}
                  className={`h-full w-full object-cover transition-opacity ${status === "ready" ? "opacity-100" : "opacity-0"}`}
                  onLoad={() => setStatus("ready")}
                  onError={() => setStatus("error")}
                />
              </div>
            </div>
          </div>

          {status === "ready" && downloadUrl && (
            <div className="mt-6 flex justify-center">
              <a
                href={downloadUrl}
                className="rounded-xl bg-neutral-900 px-8 py-3 text-sm font-bold uppercase tracking-wide text-white transition-transform hover:-translate-y-0.5"
              >
                Download AI Redesign
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
