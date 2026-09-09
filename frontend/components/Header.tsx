export default function Header() {
  return (
    <header className="relative overflow-hidden rounded-b-[28px] bg-white shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            "radial-gradient(circle at 20% 20%, #f4f4f5 0%, transparent 45%), radial-gradient(circle at 80% 0%, #eef2ff 0%, transparent 40%)",
        }}
      />
      <div className="relative mx-auto flex max-w-5xl flex-col items-center gap-3 px-6 py-14 text-center">
        <h1 className="font-display text-4xl font-bold tracking-tight text-neutral-900 sm:text-5xl">
          RoomSense
        </h1>
        <p className="text-lg text-neutral-600">Design your perfect space</p>
        <span className="mt-2 inline-flex items-center gap-2 rounded-full border-2 border-neutral-900 px-4 py-1.5 text-xs font-semibold uppercase tracking-wider text-neutral-900">
          Real Computer Vision · Real AI Redesign
        </span>
      </div>
    </header>
  );
}
