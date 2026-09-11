import Link from "next/link";

export default function Footer() {
  return (
    <footer className="mx-auto mt-10 max-w-2xl px-4 pb-6 text-center sm:px-6">
      <div className="flex items-center justify-center gap-4 text-sm text-neutral-400">
        <Link href="/terms" className="underline decoration-neutral-300 underline-offset-2 hover:text-neutral-700">
          Terms of Service
        </Link>
        <span className="text-neutral-300">•</span>
        <Link href="/privacy" className="underline decoration-neutral-300 underline-offset-2 hover:text-neutral-700">
          Privacy Policy
        </Link>
      </div>
      <p className="mt-3 text-xs text-neutral-400">
        RoomSense provides design inspiration only and is not a substitute for professional advice.
      </p>
    </footer>
  );
}
