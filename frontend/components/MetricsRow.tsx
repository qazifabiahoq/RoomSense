export default function MetricsRow({
  items,
}: {
  items: { icon: string; label: string; value: string }[];
}) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((item) => (
        <div
          key={item.label}
          className="flex flex-col items-center justify-center gap-1 rounded-xl border border-neutral-200 bg-white px-3 py-4 text-center shadow-sm"
        >
          <div className="text-lg">{item.icon}</div>
          <div className="text-[11px] font-medium uppercase tracking-wide text-neutral-400">{item.label}</div>
          <div className="font-display truncate text-sm font-semibold text-neutral-900">{item.value}</div>
        </div>
      ))}
    </div>
  );
}
