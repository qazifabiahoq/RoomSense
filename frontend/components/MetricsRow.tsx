export default function MetricsRow({
  items,
}: {
  items: { icon: string; label: string; value: string }[];
}) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {items.map((item) => (
        <div
          key={item.label}
          className="flex h-32 flex-col items-center justify-center gap-1 rounded-2xl border-t-4 border-neutral-900 bg-white p-4 text-center shadow-[0_2px_12px_rgba(0,0,0,0.08)] transition-transform hover:-translate-y-1"
        >
          <div className="text-xl">{item.icon}</div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-neutral-500">{item.label}</div>
          <div className="font-display text-lg font-bold text-neutral-900">{item.value}</div>
        </div>
      ))}
    </div>
  );
}
