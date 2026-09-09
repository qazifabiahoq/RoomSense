export default function InsightsList({ insights }: { insights: string[] }) {
  return (
    <div className="rounded-2xl border-2 border-neutral-200 bg-white p-6 sm:p-8">
      <h3 className="font-display mb-4 text-lg font-bold text-neutral-900">Insights From Your Photo</h3>
      <ul className="space-y-3">
        {insights.map((insight, i) => {
          const [title, ...rest] = insight.split(":");
          return (
            <li key={i} className="text-sm leading-relaxed text-neutral-700">
              <strong className="text-neutral-900">{title}:</strong>
              {rest.join(":")}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
