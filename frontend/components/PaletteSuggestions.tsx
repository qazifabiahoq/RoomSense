import ColorPalette from "./ColorPalette";
import { paletteSuggestions } from "@/lib/roomData";
import { RoomType } from "@/lib/types";

export default function PaletteSuggestions({ roomType }: { roomType: RoomType }) {
  const suggestions = paletteSuggestions(roomType);
  return (
    <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6 sm:p-8">
      <h3 className="font-display mb-1 text-lg font-bold text-neutral-900">
        Suggested Color Palettes for Your {roomType}
      </h3>
      <p className="mb-5 text-sm text-neutral-600">Professional color combinations that work well for this room type.</p>
      <div className="grid gap-5">
        {suggestions.map((s) => (
          <ColorPalette key={s.name} title={s.name} colors={s.colors} />
        ))}
      </div>
    </div>
  );
}
