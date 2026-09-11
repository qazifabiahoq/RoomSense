import ColorPalette from "./ColorPalette";
import { paletteSuggestions } from "@/lib/roomData";
import { RoomType } from "@/lib/types";

export default function PaletteSuggestions({ roomType }: { roomType: RoomType }) {
  const suggestions = paletteSuggestions(roomType);
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-6">
      <h3 className="font-display mb-1 text-base font-semibold text-neutral-900">Color ideas</h3>
      <p className="mb-4 text-sm text-neutral-500">A few palettes that work well for this room type.</p>
      <div className="space-y-4">
        {suggestions.map((s) => (
          <ColorPalette key={s.name} title={s.name} colors={s.colors} />
        ))}
      </div>
    </div>
  );
}
