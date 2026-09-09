"use client";

import { RoomType } from "@/lib/types";

export default function ShareBar({ roomType }: { roomType: RoomType }) {
  const url = typeof window !== "undefined" ? window.location.href : "https://roomsense.app";
  const text = `Check out my ${roomType} design from RoomSense!`;
  const encodedUrl = encodeURIComponent(url);
  const encodedText = encodeURIComponent(text);

  const links = [
    { name: "Facebook", href: `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}` },
    { name: "Twitter", href: `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}` },
    { name: "LinkedIn", href: `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}` },
    { name: "Pinterest", href: `https://pinterest.com/pin/create/button/?url=${encodedUrl}&description=${encodedText}` },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {links.map((link) => (
        <a
          key={link.name}
          href={link.href}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-lg border-2 border-neutral-900 bg-white py-3 text-center text-sm font-semibold text-neutral-900 transition-colors hover:bg-neutral-900 hover:text-white"
        >
          {link.name}
        </a>
      ))}
    </div>
  );
}
