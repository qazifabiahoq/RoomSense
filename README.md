# RoomSense

**Real computer-vision room analysis + real generative AI redesign.**

Upload a photo of your room. RoomSense detects the actual furniture and fixtures in it, measures its actual lighting and color palette, and gives you professional design recommendations — then generates a real AI redesign of the room in a style you pick.

**Live site:** _added once the frontend is deployed_
**API:** https://roomsense-vision-api.onrender.com (FastAPI, may take ~50s to wake up on first request — free tier)

---

## What's real here (and what isn't)

This app used to fake its "AI analysis" with `np.random`. It doesn't anymore. Here's exactly what each part does:

| Feature | How it works | Real or curated? |
|---|---|---|
| Furniture/fixture detection | SSDLite MobileNetV3 object detector (PyTorch, COCO-pretrained) runs on your actual uploaded photo | **Real inference**, on your image |
| Lighting rating | Measured from the actual pixel brightness of your photo | **Real measurement** |
| Color palette | K-Means clustering (scikit-learn) over your photo's actual pixels | **Real computation** |
| Detection confidence | The object detector's own average confidence score | **Real, from the model** |
| Room redesign images | Stable Diffusion (Flux) via Pollinations.ai, a free hosted generative AI API | **Real generative AI** |
| Furniture/layout recommendations | A curated knowledge base of interior-design zones, furniture, and clearances per room type | Expert-curated content, not AI-generated — this was never claimed to be ML output |
| Room type | You select it — it's not "detected" | Your input, not a model output |

Nothing in this app fabricates numbers. If a field is a "real" measurement, it comes from actually running a model or algorithm on your actual photo.

---

## Architecture

Two services, deployed separately:

- **`frontend/`** — Next.js 14 app (deploy target: **Vercel**). Handles the UI, the room-recommendation content, and proxies AI redesign requests to Pollinations.ai through a serverless API route (`/api/redesign`) so downloads work cleanly.
- **`backend/`** — FastAPI service (deploy target: **Render**). Runs the actual computer vision: object detection, brightness analysis, and color extraction. This needs a persistent Python process with real ML dependencies (PyTorch/torchvision), which is why it's on Render rather than Vercel's serverless functions.

```
Browser → Vercel (Next.js UI + /api/redesign proxy) → Render (FastAPI + PyTorch vision) 
                                                     → Pollinations.ai (Stable Diffusion)
```

### Why not just Vercel?

Vercel serverless functions are stateless and have to stay small — they're not built for a ~300MB+ PyTorch object-detection model that needs to stay loaded in memory between requests. Render runs it as a normal persistent web service instead.

---

## Running locally

**Backend:**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
# point lib/config.ts's API_BASE_URL at http://localhost:8000 for local dev
npm run dev
```

---

## Tech stack

- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS
- **Backend:** FastAPI, PyTorch + torchvision (SSDLite MobileNetV3, COCO weights), scikit-learn (K-Means), Pillow, NumPy
- **Generative AI:** Pollinations.ai (free, hosted Stable Diffusion/Flux — no API key required)
- **Hosting:** Vercel (frontend) + Render (backend)

---

## License

MIT License — free for personal and commercial use.
