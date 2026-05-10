"""
Ingredient Detection & Recipe Generation Pipeline
-------------------------------------------------
Stage 1: YOLOv8         → detect food/object bounding boxes
Stage 2: Gemini Vision  → identify ingredients from full image
Stage 3: Gemini Flash   → generate a structured recipe

Uses Google Gemini 1.5 Flash — completely FREE via Google AI Studio.
Get your key at: https://aistudio.google.com  (no credit card needed)
"""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes with labels on a copy of the image."""
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det.get("label", "food")
        conf = det.get("confidence", 0.0)
        color = (34, 197, 94)   # green
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


# ---------------------------------------------------------------------------
# Stage 1 — YOLOv8 object detection
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Wraps YOLOv8 (ultralytics).  Uses the general-purpose yolov8n.pt weights
    and keeps only detections that belong to COCO food/produce categories,
    giving a fast CPU-friendly first pass.
    """

    # COCO class IDs that are food-related
    FOOD_CLASSES = {
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
        50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 56: "chair",   # keep chair as fallback removal
    }
    # Proper food-only set (remove non-food above)
    FOOD_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.25):
        from ultralytics import YOLO
        weights = f"yolov8{model_size}.pt"
        self.model = YOLO(weights)
        self.conf = conf_threshold

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Run inference and return a list of detection dicts:
          {box: [x1,y1,x2,y2], label: str, confidence: float, crop: PIL.Image}
        """
        results = self.model(image, conf=self.conf, verbose=False)[0]
        detections = []
        h, w = image.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            label = results.names[cls_id]
            crop  = cv2_to_pil(image[y1:y2, x1:x2])

            detections.append({
                "box": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf,
                "class_id": cls_id,
                "crop": crop,
            })

        return detections


# ---------------------------------------------------------------------------
# Stage 2 — Gemini Vision ingredient identification
# ---------------------------------------------------------------------------

class GeminiVisionIdentifier:
    """
    Sends the full image to Gemini 2.5 Flash Vision to get a clean
    list of identified ingredients with confidence reasoning.
    FREE tier: 15 RPM, 1500 requests/day via Google AI Studio.
    """

    SYSTEM_PROMPT = """You are an expert culinary vision AI.
Your task is to identify ALL food ingredients visible in an image.
Be specific — distinguish between e.g. 'red bell pepper' vs 'green capsicum',
'cherry tomatoes' vs 'roma tomatoes'.
Always respond with valid JSON only — no markdown fences, no extra text."""

    USER_PROMPT = """Analyse this kitchen/ingredient image carefully.

Return a JSON object with this exact schema:
{
  "ingredients": [
    {
      "name": "ingredient name",
      "quantity_estimate": "rough estimate e.g. '2 medium' or 'handful'",
      "confidence": "high|medium|low",
      "notes": "any relevant detail e.g. 'appears ripe', 'diced'"
    }
  ],
  "scene_description": "one sentence describing the overall scene",
  "cuisine_hints": ["list", "of", "possible", "cuisine", "styles"]
}

Be thorough. Include spices, condiments, and garnishes if visible."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction=self.SYSTEM_PROMPT,
        )

    def identify(self, image: Image.Image) -> dict:
        """Send image to Gemini Vision and return parsed ingredient data."""
        response = self.model.generate_content([self.USER_PROMPT, image])
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)


# ---------------------------------------------------------------------------
# Stage 3 — Recipe generation
# ---------------------------------------------------------------------------

class RecipeGenerator:
    """Generates a structured recipe from a list of identified ingredients."""

    SYSTEM_PROMPT = """You are a world-class recipe developer with expertise in 
global cuisines, nutrition, and practical home cooking. 
You create clear, delicious, achievable recipes.
Always respond with valid JSON only — no markdown, no preamble."""

    def build_user_prompt(
        self,
        ingredients: list[dict],
        servings: int,
        dietary_prefs: list[str],
        cuisine_hints: list[str],
    ) -> str:
        ing_list = "\n".join(
            f"- {i['name']} ({i.get('quantity_estimate','some')})"
            for i in ingredients
        )
        prefs = ", ".join(dietary_prefs) if dietary_prefs else "none"
        hints = ", ".join(cuisine_hints) if cuisine_hints else "any"

        return f"""Available ingredients:
{ing_list}

Constraints:
- Servings: {servings}
- Dietary preferences/restrictions: {prefs}
- Cuisine style hints from image: {hints}

Generate a complete recipe using PRIMARILY these ingredients (you may assume 
basic pantry staples like salt, pepper, oil, water are available).

Return JSON with this exact schema:
{{
  "recipe_name": "...",
  "cuisine": "...",
  "difficulty": "Easy|Medium|Hard",
  "prep_time_minutes": 0,
  "cook_time_minutes": 0,
  "servings": {servings},
  "description": "2-3 sentence appetising description",
  "ingredients": [
    {{"item": "...", "amount": "...", "unit": "...", "notes": "optional prep note"}}
  ],
  "instructions": [
    {{"step": 1, "title": "short title", "detail": "full instruction text", "time_minutes": 0}}
  ],
  "nutrition_per_serving": {{
    "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fibre_g": 0
  }},
  "tips": ["tip 1", "tip 2"],
  "substitutions": ["substitution suggestion 1", "substitution suggestion 2"]
}}"""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction=self.SYSTEM_PROMPT,
        )

    def generate(
        self,
        ingredients: list[dict],
        servings: int = 2,
        dietary_prefs: Optional[list[str]] = None,
        cuisine_hints: Optional[list[str]] = None,
    ) -> dict:
        prompt = self.build_user_prompt(
            ingredients,
            servings,
            dietary_prefs or [],
            cuisine_hints or [],
        )
        response = self.model.generate_content(prompt)
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)


# ---------------------------------------------------------------------------
# Orchestrator — ties all three stages together
# ---------------------------------------------------------------------------

class IngredientToRecipePipeline:
    """
    Full pipeline:
      image → YOLOv8 detection → Gemini Vision ID → Recipe generation
    Uses Google Gemini 2.5 Flash (FREE) for both vision and text stages.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        yolo_model_size: str = "n",
        yolo_conf: float = 0.25,
    ):
        self.detector   = YOLODetector(model_size=yolo_model_size, conf_threshold=yolo_conf)
        self.identifier = GeminiVisionIdentifier(api_key=api_key)
        self.generator  = RecipeGenerator(api_key=api_key)

    def run(
        self,
        image_input,          # PIL.Image or np.ndarray or file path
        servings: int = 2,
        dietary_prefs: Optional[list[str]] = None,
        cuisine_hints_override: Optional[list[str]] = None,
        progress_callback=None,
    ) -> dict:
        """
        Returns a dict with keys:
          detections, annotated_image, vision_data, recipe
        """
        def _progress(msg):
            if progress_callback:
                progress_callback(msg)

        # --- Normalise input ---
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = cv2_to_pil(image_input)
        else:
            pil_img = image_input.convert("RGB")

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # --- Stage 1: YOLO ---
        _progress("🔍 Stage 1/3 — Running YOLOv8 object detection…")
        detections = self.detector.detect(cv_img)
        annotated  = draw_boxes(cv_img, detections)
        annotated_pil = cv2_to_pil(annotated)

        # --- Stage 2: Gemini Vision ---
        _progress("👁️ Stage 2/3 — Gemini Vision identifying ingredients…")
        vision_data = self.identifier.identify(pil_img)

        # Merge cuisine hints
        hints = cuisine_hints_override or vision_data.get("cuisine_hints", [])

        # --- Stage 3: Recipe ---
        _progress("🍳 Stage 3/3 — Generating recipe…")
        recipe = self.generator.generate(
            ingredients=vision_data["ingredients"],
            servings=servings,
            dietary_prefs=dietary_prefs,
            cuisine_hints=hints,
        )

        _progress("✅ Pipeline complete!")
        return {
            "detections": detections,
            "annotated_image": annotated_pil,
            "vision_data": vision_data,
            "recipe": recipe,
        }