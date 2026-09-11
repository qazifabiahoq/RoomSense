# RoomSense

AI-Powered Room Design. A Full Plan From One Photo.

Live Demo: [https://room-sense-inky.vercel.app](https://room-sense-inky.vercel.app)

API Health Check: [https://roomsense-vision-api.onrender.com/health](https://roomsense-vision-api.onrender.com/health)

---

## The Problem

Redesigning a room usually comes down to two bad options. You hire an interior designer, which is expensive and slow, often weeks of back and forth before you see a single mockup. Or you scroll saved pins and mood boards that look nothing like your actual room, guessing at what might fit your space, your furniture, your lighting.

Generic design advice does not know what is already in your room. It does not know whether your light is bright and natural or dim and artificial, whether your walls are already crowded with furniture, or what colors are already dominating the space. Without that, "add a floor lamp here" is just a guess dressed up as advice.

RoomSense was built to remove the guessing. It looks at your actual room and works from there.

---

## What RoomSense Does

You upload a photo of a room. A computer vision model looks at the image and finds the furniture and fixtures actually in it: sofas, chairs, tables, lamps, whatever is really there. Separately, the system measures how bright the room actually is and extracts the dominant colors from the photo's own pixels, not a generic palette pulled from a room-type template.

With real information about your space in hand, RoomSense returns a design plan: furniture zones, placement guidance, lighting setup, and clearances, written for the room type you selected. From there you can generate a real AI redesign of your room in a chosen style using generative image AI, and download the result.

Nothing about the analysis is invented. If the app reports four items found or shows a color swatch, that came from a model actually looking at your photo, not a placeholder standing in for one.

---

## The Vision Pipeline

This is the part that does real work on your actual photo, and it runs as three separate steps.

**Object detection.** An SSDLite MobileNetV3 model, pretrained on the COCO dataset, runs inference on the uploaded image and returns bounding boxes for whatever furniture and fixtures it recognizes: couches, chairs, beds, dining tables, TVs, and dozens of other household categories. SSDLite MobileNetV3 was chosen deliberately over a heavier detector because it is built for exactly this constraint: real-time inference on modest CPU hardware, which is what a low-cost hosted backend actually has to work with. The detected boxes are drawn directly onto your photo in the interface, so you can see exactly what the model found and how confident it was about each item.

**Lighting measurement.** The backend reads the actual pixel brightness of your photo and classifies the lighting into a plain-language rating, from low light to natural and excellent. This is a direct measurement of your image, not an assumption based on room type or time of day.

**Color extraction.** K-Means clustering runs over the photo's real pixels to pull out the room's actual dominant colors as a palette. It is unsupervised learning applied to your specific photo, not a stock palette assigned because you picked "Living Room" from a dropdown.

The room type itself is the one input that is not detected. You select it, because guessing whether a photographed room is a bedroom or a home office is a much harder and lower-value problem than analyzing what is inside it, and there was no reason to fake a prediction there when a dropdown does the job honestly.

---

## Why Two Separate Services

RoomSense is split into a Next.js frontend on Vercel and a FastAPI backend on Render, deployed and scaled independently.

The reason is the object detection model itself. It is a real PyTorch model that needs to sit loaded in memory across requests, ready to run inference the moment a photo comes in. That is fundamentally a persistent process, not a short-lived function invocation. Vercel's serverless functions are built for exactly the opposite shape of workload: fast, stateless, and cold-started on every call. Trying to force a few hundred megabytes of ML model into that model would mean reloading it on every single request, which is slow, wasteful, and eventually just does not work within serverless memory and time limits.

So the vision pipeline runs on Render as an always-on FastAPI service instead, while Vercel handles the interface, the design recommendation content, and a lightweight serverless route that proxies AI image generation requests. Each half runs on the infrastructure actually suited to it.

---

## Honesty About What's Real

An earlier version of this project faked its "AI analysis" with random number generation. Room type, confidence score, detected furniture, all of it was `np.random` dressed up to look like a model's output. It does not do that anymore, and this section exists because that history is worth being upfront about.

Every field the app shows you now falls into one of two honest categories. It is either a real measurement or model output computed from your actual photo (furniture detection, lighting rating, color palette, and the detector's own confidence scores all fall here), or it is clearly curated content that was never claimed to be AI-generated in the first place (the furniture and layout recommendations come from a hand-built interior design knowledge base, not a model). The one AI-generated visual output, the redesigned room image, is real generative AI, produced by an actual Stable Diffusion model through Pollinations.ai's free hosted API.

Nothing in the current version fabricates a number to look more impressive than what the system actually did.

---

## Who This Is Built For

Homeowners and renters who want a real plan for a room before spending money on furniture, without paying for a designer to tell them what they could see for themselves with the right tools. First-time buyers and anyone moving into a new place who want to walk in with a layout already figured out instead of guessing on move-in day. People who just want to see their own room reimagined in a different style before committing to paint, furniture, or a full renovation.

---

## Technical Stack

The frontend is a Next.js 14 application written in TypeScript, styled with Tailwind CSS, and deployed on Vercel. It handles the photo upload interface, renders the live detection overlay on top of your image, hosts the curated design recommendation content, and exposes a serverless API route that proxies AI redesign requests so downloads work cleanly with proper filenames.

The backend is a FastAPI service written in Python and deployed on Render. It loads the SSDLite MobileNetV3 object detection model from PyTorch and torchvision at startup, runs brightness analysis with NumPy and Pillow, and extracts color palettes with scikit-learn's K-Means implementation. It exposes a single analysis endpoint that a photo is posted to and a structured JSON response, containing detections, lighting, and color data, comes back.

The AI room redesign feature calls Pollinations.ai, a free hosted Stable Diffusion endpoint that requires no API key, through the Vercel serverless proxy.

---

## The Bigger Picture

Most "AI-powered" home design tools on the market are either a lookup table wearing an AI label, or a real model bolted onto marketing copy that oversells what it actually does. RoomSense was rebuilt specifically to not be either of those things: the parts that claim to be real computer vision actually run inference on your photo, and the parts that are curated design knowledge are labeled as exactly that instead of dressed up as machine intelligence.

The result is a smaller, more honest set of claims. It just happens that all of them are true.
