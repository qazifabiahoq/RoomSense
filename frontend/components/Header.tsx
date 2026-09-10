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
        <div className="mb-4 flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-500">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M3 11L12 4l9 7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M5 10v9a1 1 0 001 1h4v-6h4v6h4a1 1 0 001-1v-9" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="font-display text-xl font-bold tracking-tight text-white">RoomSense</span>
        </div>
        <h1 className="font-display max-w-xl text-3xl font-bold leading-tight text-white sm:text-4xl">
          Design your room with a single photo
        </h1>
        <p className="mx-auto mt-3 max-w-md text-base text-white/85 sm:text-lg">
          Upload a photo and get furniture ideas, color palettes, and a full design plan in seconds.
        </p>
      </div>
    </header>
  );
}
