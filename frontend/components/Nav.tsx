export default function Nav() {
  return (
    <nav className="sticky top-0 z-10 border-b border-neutral-200/70 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center gap-2 px-5 py-3">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-500">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <path d="M3 11L12 4l9 7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path
              d="M5 10v9a1 1 0 001 1h4v-6h4v6h4a1 1 0 001-1v-9"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <span className="font-display text-base font-bold tracking-tight text-neutral-900">RoomSense</span>
      </div>
    </nav>
  );
}
