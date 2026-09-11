export default function InsightsList({ insights }: { insights: string[] }) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-6">
      <h3 className="font-display mb-3 text-base font-semibold text-neutral-900">A few tips for your space</h3>
      <ul className="space-y-2.5">
        {insights.map((insight, i) => {
          const [title, ...rest] = insight.split(":");
          return (
            <li key={i} className="text-sm leading-relaxed text-neutral-600">
              <span className="font-medium text-neutral-900">{title}.</span>
              {rest.join(":")}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
