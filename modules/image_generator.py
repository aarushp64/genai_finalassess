"""
NeuroTutor AI - AI Concept Illustration Generator
Author: Aarush Parate (Roll No: 1)
Unit IV: Generative AI Tools and Techniques
Converts AI concepts into rich educational diffusion prompts and fetches illustrations via hosted diffusion API.
"""

import urllib.parse
import requests
import io
import random
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Tuple


class ImageGenerator:
    """Generates high-resolution educational concept diagrams via hosted diffusion inference."""

    @staticmethod
    def enhance_concept_prompt(concept: str) -> str:
        """
        Expands a raw concept into a high-fidelity educational schematic prompt for diffusion models.
        """
        clean_concept = concept.strip()
        enhanced = (
            f"A clean, modern, professional educational infographic diagram illustrating '{clean_concept}'. "
            f"High resolution 4k, technical vector blueprint style, deep navy and steel grey background, "
            f"muted cyan and blue glowing accents, neat labeled boxes, mathematical directional arrows, "
            f"clear scientific typography, educational poster design, trending on ArtStation."
        )
        return enhanced

    @staticmethod
    def generate_illustration(concept: str, seed: int = 42) -> Dict[str, Any]:
        """
        Queries a hosted diffusion inference API to generate the educational illustration.
        Includes a fallback procedural vector graphic generator in case of network outages.
        """
        enhanced_prompt = ImageGenerator.enhance_concept_prompt(concept)
        encoded_prompt = urllib.parse.quote(enhanced_prompt)

        # Hosted Diffusion Endpoint (Pollinations AI free generative diffusion inference)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=512&nologo=true&seed={seed}"

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and len(response.content) > 5000:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                return {
                    "image_bytes": image_bytes,
                    "image": image,
                    "prompt_used": enhanced_prompt,
                    "provider": "Hosted Diffusion Model (Pollinations AI Inference Endpoint)",
                    "status": "success",
                }
            else:
                print(f"[ImageGenerator] Diffusion API returned status {response.status_code}. Using local graphic generator...")
        except Exception as e:
            print(f"[ImageGenerator] Network error contacting diffusion API: {e}. Using local graphic generator...")

        # Fallback: Procedurally generated educational diagram
        fallback_img, fallback_bytes = ImageGenerator._create_procedural_fallback_image(concept)
        return {
            "image_bytes": fallback_bytes,
            "image": fallback_img,
            "prompt_used": enhanced_prompt,
            "provider": "NeuroTutor Procedural Infographic Generator (Offline Fallback)",
            "status": "fallback",
        }

    @staticmethod
    def _create_procedural_fallback_image(concept: str) -> Tuple[Image.Image, bytes]:
        """Creates a high-contrast educational schematic card using PIL."""
        width, height = 768, 512
        img = Image.new("RGB", (width, height), color="#171D29")
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([20, 20, width - 20, height - 20], outline="#39465A", width=2)

        # Draw decorative diagram blocks
        colors = ["#293548", "#303E52", "#293548"]
        box_coords = [
            (80, 180, 220, 320),
            (310, 150, 460, 350),
            (550, 180, 690, 320),
        ]
        labels = ["Input Layer\nx ~ p(x)", "Processing Core\nTransformer / NN", "Output Layer\nPrediction y"]

        for idx, (coord, label) in enumerate(zip(box_coords, labels)):
            draw.rounded_rectangle(coord, radius=8, fill=colors[idx % len(colors)], outline="#8BAFD0", width=2)
            draw.text((coord[0] + 15, coord[1] + 45), label, fill="#F2F4F7")

        # Draw connecting arrows
        draw.line([(220, 250), (310, 250)], fill="#8BAFD0", width=4)
        draw.line([(460, 250), (550, 250)], fill="#8BAFD0", width=4)

        # Title
        draw.text((80, 60), f"NeuroTutor AI — Concept Schematic: {concept.upper()}", fill="#F2F4F7")
        draw.text((80, 95), "Educational Architecture Representation & Dataflow Pipeline", fill="#AAB5C4")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return img, buf.getvalue()
