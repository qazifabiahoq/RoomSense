"""AI room redesign via Pollinations.ai (real Stable Diffusion generation, free, no key needed)."""
import io
from typing import Dict

import requests
import urllib.parse
from PIL import Image

STYLES: Dict[str, Dict] = {
    "Modern Minimalist": {
        "description": "Clean Scandinavian aesthetic with sleek furniture, neutral tones, and open spaces",
        "colors": ["#FFFFFF", "#F5F5F5", "#E0E0E0", "#757575"],
        "prompt": "modern minimalist interior design, sleek contemporary furniture, clean white walls, "
                  "scandinavian style, bright natural light, open space, professional interior photography, 8k uhd",
    },
    "Cozy Traditional": {
        "description": "Warm, inviting spaces with classic furniture, rich textures, and comfortable seating",
        "colors": ["#8B4513", "#D2691E", "#DEB887", "#F5DEB3"],
        "prompt": "cozy traditional interior design, classic comfortable furniture, warm wood tones, soft textiles, "
                  "warm lighting, inviting atmosphere, professional interior photography, 8k uhd",
    },
}


def redesign_room(style: str, room_type: str) -> Image.Image:
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style}")

    style_config = STYLES[style]
    full_prompt = f"{room_type}, {style_config['prompt']}"
    encoded_prompt = urllib.parse.quote(full_prompt)

    request_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=768&height=768&model=flux&nologo=true&enhance=true"
    )

    response = requests.get(request_url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")
