import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoomSense - Intelligent Space Planning",
  description: "Upload a photo of your room and get real AI-powered layout, lighting, and furniture recommendations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
