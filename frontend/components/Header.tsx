export default function Header() {
  return (
    <header className="relative overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?w=1600&h=900&fit=crop&q=80"
        alt="A beautifully designed living room"
        className="absolute inset-0 h-full w-full object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-neutral-900/80 via-neutral-900/55 to-neutral-900/30" />

      <div className="relative mx-auto flex max-w-5xl flex-col items-center px-5 py-16 text-center sm:py-24">
        <span className="mb-4 inline-flex items-center gap-1.5 rounded-full bg-white/15 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white ring-1 ring-inset ring-white/25">
          Real computer vision, not a guess
        </span>
        <h1 className="font-display max-w-xl text-4xl font-bold leading-tight text-white sm:text-5xl">
          Design your room with a single photo
        </h1>
        <p className="mx-auto mt-3 max-w-md text-base text-white/85 sm:text-lg">
          Upload a photo and get furniture ideas, color palettes, and a full design plan in seconds.
        </p>
      </div>
    </header>
  );
}
