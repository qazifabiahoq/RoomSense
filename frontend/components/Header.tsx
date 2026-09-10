export default function Header() {
  return (
    <header className="border-b border-neutral-200 bg-white">
      <div className="mx-auto max-w-5xl px-5 py-10 text-center sm:py-14">
        <h1 className="font-display text-3xl font-bold tracking-tight text-neutral-900 sm:text-4xl">
          RoomSense
        </h1>
        <p className="mx-auto mt-3 max-w-md text-base text-neutral-500 sm:text-lg">
          Upload a photo of your room and get a personalized design plan in seconds.
        </p>
      </div>
    </header>
  );
}
