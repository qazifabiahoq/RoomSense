"use client";

import { ROOM_TYPES } from "@/lib/roomData";
import { RoomType } from "@/lib/types";

export default function RoomPicker({
  value,
  onChange,
}: {
  value: RoomType;
  onChange: (value: RoomType) => void;
}) {
  return (
    <div>
      <p className="mb-3 text-sm font-semibold text-neutral-900">What room are you designing?</p>
      <div className="flex flex-wrap gap-2">
        {ROOM_TYPES.map((room) => {
          const active = room === value;
          return (
            <button
              key={room}
              type="button"
              onClick={() => onChange(room)}
              className={`rounded-full border-2 px-4 py-2 text-sm font-medium transition-all ${
                active
                  ? "border-neutral-900 bg-neutral-900 text-white shadow-md"
                  : "border-neutral-200 bg-white text-neutral-700 hover:border-neutral-400"
              }`}
            >
              {room}
            </button>
          );
        })}
      </div>
    </div>
  );
}
