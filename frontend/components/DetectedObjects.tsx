import { Detection } from "@/lib/types";

export default function DetectedObjects({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return (
      <p className="text-sm text-neutral-500">
        No furniture or fixtures detected with high confidence in this photo.
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-2">
      {detections.map((d, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1.5 rounded-full border-2 border-neutral-900 bg-white px-3 py-1.5 text-sm font-semibold text-neutral-900"
        >
          {d.label}
          <span className="text-xs font-normal text-neutral-500">{Math.round(d.confidence * 100)}%</span>
        </span>
      ))}
    </div>
  );
}
