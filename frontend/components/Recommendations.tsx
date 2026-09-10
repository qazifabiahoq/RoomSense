import { ROOM_CONFIGS } from "@/lib/roomData";
import { RoomType } from "@/lib/types";

export default function Recommendations({ roomType }: { roomType: RoomType }) {
  const zones = ROOM_CONFIGS[roomType];

  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-6">
      <h2 className="font-display mb-1 text-lg font-semibold text-neutral-900">
        Design plan for your {roomType.toLowerCase()}
      </h2>
      <p className="mb-5 text-sm text-neutral-500">
        Furniture, layout, and lighting guidance based on interior design best practices.
      </p>
      <div className="space-y-4">
        {zones.map((zone) => (
          <div key={zone.name} className="rounded-lg border border-neutral-200 bg-neutral-50/60 p-4">
            <h3 className="font-display mb-1.5 text-base font-semibold text-neutral-900">{zone.name}</h3>
            <p className="mb-3 text-sm text-neutral-600">
              <span className="font-medium text-neutral-800">Where:</span> {zone.location}
              <br />
              <span className="font-medium text-neutral-800">Lighting:</span> {zone.lighting}
            </p>
            <div className="rounded-lg border border-neutral-200 bg-white p-3">
              <p className="mb-1 text-xs font-medium uppercase tracking-wide text-neutral-400">Furniture</p>
              {zone.furniture.map((item) => (
                <div key={item} className="border-b border-neutral-100 py-1 text-sm text-neutral-700 last:border-0">
                  {item}
                </div>
              ))}
            </div>
            <div className="mt-3 text-sm text-neutral-600">
              <span className="font-medium text-neutral-800">Keep in mind:</span>
              {zone.considerations.map((c) => (
                <div key={c}>{c}</div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
