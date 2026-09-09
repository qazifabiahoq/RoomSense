import { ROOM_CONFIGS } from "@/lib/roomData";
import { RoomType } from "@/lib/types";

export default function Recommendations({ roomType }: { roomType: RoomType }) {
  const zones = ROOM_CONFIGS[roomType];

  return (
    <div className="rounded-[20px] border-2 border-neutral-200 bg-white p-6 sm:p-10">
      <h2 className="font-display mb-2 text-2xl font-bold text-neutral-900">
        Smart Recommendations for Your {roomType}
      </h2>
      <p className="mb-6 text-neutral-600">
        Professional design guidance based on established interior design principles.
      </p>
      <div className="grid gap-5">
        {zones.map((zone) => (
          <div
            key={zone.name}
            className="rounded-2xl border-2 border-neutral-200 bg-neutral-50 p-6 transition-transform hover:translate-x-1"
          >
            <h3 className="font-display mb-2 text-xl font-bold text-neutral-900">{zone.name}</h3>
            <p className="mb-3 text-sm text-neutral-700">
              <strong>Optimal Location:</strong> {zone.location}
              <br />
              <strong>Lighting Setup:</strong> {zone.lighting}
            </p>
            <div className="rounded-xl border border-neutral-200 bg-white p-4">
              <p className="mb-1 text-sm font-semibold text-neutral-900">Recommended Furniture</p>
              {zone.furniture.map((item) => (
                <div key={item} className="border-b border-neutral-100 py-1.5 text-sm text-neutral-800 last:border-0">
                  • {item}
                </div>
              ))}
            </div>
            <div className="mt-4 text-sm text-neutral-700">
              <strong>Key Considerations:</strong>
              {zone.considerations.map((c) => (
                <div key={c}>• {c}</div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
