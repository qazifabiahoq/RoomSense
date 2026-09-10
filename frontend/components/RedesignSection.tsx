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
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-6">
      <h2 className="font-display mb-1 text-lg font-semibold text-neutral-900">See it in a new style</h2>
      <p className="mb-5 text-sm text-neutral-500">
        Pick a style and we will generate a redesigned version of your room.
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        {Object.entries(REDESIGN_STYLES).map(([name, style]) => (
          <button
            key={name}
            onClick={() => selectStyle(name)}
            className={`rounded-lg border p-4 text-left transition-colors ${
              selectedStyle === name
                ? "border-neutral-900 bg-neutral-50"
                : "border-neutral-200 bg-white hover:border-neutral-300"
            }`}
          >
            <p className="font-display text-sm font-semibold text-neutral-900">{name}</p>
            <p className="mb-2.5 mt-0.5 text-xs text-neutral-500">{style.description}</p>
            <div className="flex gap-1">
              {style.colors.map((c) => (
                <span key={c} className="h-4 w-4 rounded-full border border-neutral-200" style={{ backgroundColor: c }} />
              ))}
            </div>
          </button>
        ))}
      </div>

      {selectedStyle && redesignUrl && (
        <div className="mt-6">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="mb-2 text-center text-xs font-medium uppercase tracking-wide text-neutral-400">Before</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={originalImageUrl} alt="Your room" className="w-full rounded-lg border border-neutral-200" />
            </div>
            <div>
              <p className="mb-2 text-center text-xs font-medium uppercase tracking-wide text-neutral-400">
                After: {selectedStyle}
              </p>
              <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-neutral-200 bg-neutral-50">
                {status === "loading" && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-neutral-400">
                    <div className="h-6 w-6 animate-spin rounded-full border-2 border-neutral-200 border-t-neutral-600" />
                    <p className="text-xs">Generating, usually takes 20 to 30 seconds</p>
                  </div>
                )}
                {status === "error" && (
                  <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-red-600">
                    Something went wrong. Try again or pick another style.
                  </div>
                )}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  key={imgKey}
                  src={redesignUrl}
                  alt={`${roomType} redesigned in ${selectedStyle} style`}
                  className={`h-full w-full object-cover transition-opacity ${status === "ready" ? "opacity-100" : "opacity-0"}`}
                  onLoad={() => setStatus("ready")}
                  onError={() => setStatus("error")}
                />
              </div>
            </div>
          </div>

          {status === "ready" && downloadUrl && (
            <div className="mt-5 flex justify-center">
              <a
                href={downloadUrl}
                className="rounded-lg bg-neutral-900 px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-neutral-700"
              >
                Download this design
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
