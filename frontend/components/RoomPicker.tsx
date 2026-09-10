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
      <p className="mb-3 text-sm font-medium text-neutral-700">Room type</p>
      <div className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {ROOM_TYPES.map((room) => {
          const active = room === value;
          return (
            <button
              key={room}
              type="button"
              onClick={() => onChange(room)}
              className={`shrink-0 rounded-lg border px-3.5 py-2 text-sm font-medium transition-colors ${
                active
                  ? "border-neutral-900 bg-neutral-900 text-white"
                  : "border-neutral-200 bg-white text-neutral-600 hover:border-neutral-300 hover:text-neutral-900"
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
