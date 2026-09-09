import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

// Proxies to Pollinations.ai, a free hosted Stable Diffusion (Flux) endpoint.
// Proxying (instead of calling it client-side) lets us set a real download
// filename/Content-Disposition and avoids CORS issues on the client canvas.
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const prompt = searchParams.get("prompt");
  const download = searchParams.get("download") === "1";
  const filename = searchParams.get("filename") || "roomsense-redesign.png";

  if (!prompt) {
    return new Response(JSON.stringify({ error: "Missing prompt" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const upstreamUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(
    prompt
  )}?width=768&height=768&model=flux&nologo=true&enhance=true`;

  try {
    const upstream = await fetch(upstreamUrl, {
      signal: AbortSignal.timeout(55000),
    });

    if (!upstream.ok || !upstream.body) {
      return new Response(
        JSON.stringify({ error: `Image generation failed (status ${upstream.status})` }),
        { status: 502, headers: { "Content-Type": "application/json" } }
      );
    }

    const headers = new Headers();
    headers.set("Content-Type", upstream.headers.get("content-type") || "image/png");
    headers.set("Cache-Control", "no-store");
    if (download) {
      headers.set("Content-Disposition", `attachment; filename="${filename}"`);
    }

    return new Response(upstream.body, { status: 200, headers });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Image generation timed out or failed. Please try again." }),
      { status: 504, headers: { "Content-Type": "application/json" } }
    );
  }
}
