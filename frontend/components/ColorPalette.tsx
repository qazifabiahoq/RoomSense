export default function ColorPalette({ colors, title }: { colors: string[]; title?: string }) {
  return (
    <div>
      {title && <p className="mb-2 text-sm font-semibold text-neutral-900">{title}</p>}
      <div className="flex gap-3">
        {colors.map((color, i) => (
          <div
            key={`${color}-${i}`}
            className="flex h-16 flex-1 items-end justify-center rounded-lg border-2 border-neutral-200 pb-1.5 shadow-sm"
            style={{ backgroundColor: color }}
          >
            <span className="rounded bg-white/90 px-1.5 py-0.5 text-[10px] font-semibold text-neutral-900">
              {color}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
