import { Detection } from "@/lib/types";

export default function DetectedObjects({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return (
      <p className="text-sm text-neutral-400">
        We could not confidently identify furniture in this photo. Try a wider or brighter shot.
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {detections.map((d, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1.5 rounded-lg border border-neutral-200 bg-neutral-50 px-2.5 py-1 text-sm font-medium text-neutral-700"
        >
          {d.label}
          <span className="text-xs text-neutral-400">{Math.round(d.confidence * 100)}%</span>
        </span>
      ))}
    </div>
  );
}
