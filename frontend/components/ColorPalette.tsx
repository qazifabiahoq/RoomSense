export default function ColorPalette({ colors, title }: { colors: string[]; title?: string }) {
  return (
    <div>
      {title && <h3 className="font-display mb-3 text-base font-semibold text-neutral-900">{title}</h3>}
      <div className="flex gap-2">
        {colors.map((color, i) => (
          <div
            key={`${color}-${i}`}
            className="flex h-14 flex-1 items-end justify-center rounded-lg border border-neutral-200 pb-1"
            style={{ backgroundColor: color }}
          >
            <span className="rounded bg-white/85 px-1.5 py-0.5 text-[10px] font-medium text-neutral-700">
              {color}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
