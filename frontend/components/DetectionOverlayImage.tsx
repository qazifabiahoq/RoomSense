"use client";

import { Detection } from "@/lib/types";

export default function DetectionOverlayImage({
  src,
  detections,
  alt,
}: {
  src: string;
  detections: Detection[];
  alt: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl border-2 border-neutral-200 bg-neutral-50">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="block w-full" />
      {detections.map((d, i) => {
        const [xmin, ymin, xmax, ymax] = d.box;
        return (
          <div
            key={i}
            className="absolute border-2 border-emerald-400/90 shadow-[0_0_0_1px_rgba(0,0,0,0.15)]"
            style={{
              left: `${xmin * 100}%`,
              top: `${ymin * 100}%`,
              width: `${(xmax - xmin) * 100}%`,
              height: `${(ymax - ymin) * 100}%`,
            }}
          >
            <span className="absolute -top-6 left-0 whitespace-nowrap rounded-t-md bg-emerald-500 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
              {d.label} {Math.round(d.confidence * 100)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
