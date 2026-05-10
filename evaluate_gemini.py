"""
Gemini Ingredient Identification — Evaluation on Freiburg Groceries Dataset
============================================================================

Measures Top-1 and Top-3 accuracy of Gemini Vision's ingredient identification
against the ground truth labels provided by the Freiburg Groceries Dataset.

SETUP:
1. Download the dataset from:
   http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/
2. Unzip it — you'll get a folder structure like:
      freiburg_groceries_dataset/
          images/
              Apple/
                  00000000.png
                  00000001.png
                  ...
              Banana/
              Butter/
              ... (25 classes total)
3. Set DATASET_PATH below to point at that folder.
4. Run:  python evaluate_gemini.py

OUTPUT:
- Prints per-class accuracy and overall Top-1 / Top-3 accuracy
- Saves full results to evaluation_results.json
- Saves a summary CSV to evaluation_summary.csv
"""

import os
import sys
import json
import time
import random
import csv
from pathlib import Path
from dotenv import load_dotenv

# ── Config — edit these ───────────────────────────────────────────────────────

DATASET_PATH   = "/Users/vinayakgoswamy/Downloads/images"  # path to the images/ folder
IMAGES_PER_CLASS = 5    # how many images to test per class (max ~80 per class)
RANDOM_SEED    = 42        # for reproducibility
DELAY_SECONDS  = 7         # pause between API calls to stay within free tier rate limit

# ─────────────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import GeminiVisionIdentifier
from PIL import Image


# ── Label mapping ─────────────────────────────────────────────────────────────
# Maps Freiburg folder names → list of acceptable Gemini match strings
# Gemini returns natural language so we allow several valid synonyms per class

LABEL_MAP = {
    "Apple":          ["apple"],
    "Banana":         ["banana"],
    "Butter":         ["butter"],
    "Candy":          ["candy", "sweets", "confectionery", "chocolate"],
    "Cereal":         ["cereal", "oats", "granola", "muesli"],
    "Chips":          ["chips", "crisps", "crackers"],
    "Chocolate":      ["chocolate", "cocoa"],
    "Coffee":         ["coffee"],
    "Corn":           ["corn", "maize", "sweetcorn"],
    "Fish":           ["fish", "tuna", "salmon", "sardine", "cod"],
    "Flour":          ["flour"],
    "Honey":          ["honey"],
    "Jam":            ["jam", "jelly", "marmalade", "preserve"],
    "Juice":          ["juice"],
    "Ketchup":        ["ketchup", "tomato sauce", "tomato ketchup"],
    "Lemon":          ["lemon", "lime", "citrus"],
    "Milk":           ["milk"],
    "Mustard":        ["mustard"],
    "Oil":            ["oil", "olive oil", "vegetable oil", "cooking oil"],
    "Orange":         ["orange"],
    "Pasta":          ["pasta", "spaghetti", "noodle", "penne", "fusilli"],
    "Rice":           ["rice"],
    "Soda":           ["soda", "soft drink", "cola", "lemonade", "fizzy"],
    "Tomato":         ["tomato"],
    "Vinegar":        ["vinegar"],
    "Water":          ["water"],
    "Wine":           ["wine"],
    "Yogurt":         ["yogurt", "yoghurt"],
}


def is_match(ground_truth_class: str, identified_ingredients: list[dict]) -> bool:
    """
    Returns True if any identified ingredient contains one of the
    acceptable label strings for the ground truth class.
    Case-insensitive substring match.
    """
    acceptable = LABEL_MAP.get(ground_truth_class, [ground_truth_class.lower()])
    all_names  = " ".join(i["name"].lower() for i in identified_ingredients)
    return any(term in all_names for term in acceptable)


def is_top3_match(ground_truth_class: str, identified_ingredients: list[dict]) -> bool:
    """
    Top-3: checks only the first 3 (highest-confidence) identified ingredients.
    """
    acceptable = LABEL_MAP.get(ground_truth_class, [ground_truth_class.lower()])
    top3_names = " ".join(i["name"].lower() for i in identified_ingredients[:3])
    return any(term in top3_names for term in acceptable)


def load_image_paths(dataset_path: str, images_per_class: int, seed: int) -> list[dict]:
    """
    Walks the dataset directory and returns a list of
    {path, class_name} dicts, randomly sampled per class.
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {dataset_dir.resolve()}\n"
            "   Download it from: http://aisdatasets.informatik.uni-freiburg.de/"
            "freiburg_groceries_dataset/\n"
            "   Then set DATASET_PATH in this script."
        )

    rng     = random.Random(seed)
    samples = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images     = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        if not images:
            continue
        chosen = rng.sample(images, min(images_per_class, len(images)))
        for img_path in chosen:
            samples.append({"path": img_path, "class": class_name})

    return samples


def run_evaluation():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("❌  GEMINI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("🔍 Loading dataset...")
    samples = load_image_paths(DATASET_PATH, IMAGES_PER_CLASS, RANDOM_SEED)
    print(f"   {len(samples)} images across {len(set(s['class'] for s in samples))} classes\n")

    identifier = GeminiVisionIdentifier(api_key=api_key)

    results    = []          # one dict per image
    per_class  = {}          # class_name → {correct_top1, correct_top3, total}

    print(f"{'#':<5} {'Class':<15} {'GT Match?':<12} {'Identified (top 3)'}")
    print("-" * 72)

    for i, sample in enumerate(samples):
        class_name = sample["class"]
        img_path   = sample["path"]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Call Gemini — auto-retry on 429 rate limit errors
        max_retries = 3
        ingredients = []
        error_msg   = None

        for attempt in range(max_retries):
            try:
                vision_data = identifier.identify(image)
                ingredients = vision_data.get("ingredients", [])
                error_msg   = None
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait = 60 if attempt == 0 else 90
                    print(f"  ⚠️  Rate limit hit — waiting {wait}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                else:
                    error_msg = error_str
                    print(f"  [{i+1}/{len(samples)}] {class_name:<15} ERROR: {error_str}")
                    break

        if error_msg or (not ingredients and error_msg is not None):
            results.append({
                "index": i + 1,
                "class": class_name,
                "image": str(img_path),
                "top1_correct": False,
                "top3_correct": False,
                "identified": [],
                "error": error_msg,
            })
            time.sleep(DELAY_SECONDS)
            continue

        top1 = is_match(class_name, ingredients)
        top3 = is_top3_match(class_name, ingredients)

        # Per-class tracking
        if class_name not in per_class:
            per_class[class_name] = {"top1": 0, "top3": 0, "total": 0}
        per_class[class_name]["total"] += 1
        if top1:
            per_class[class_name]["top1"] += 1
        if top3:
            per_class[class_name]["top3"] += 1

        # Display
        top3_names = ", ".join(ing["name"] for ing in ingredients[:3])
        status     = "✅ Yes" if top1 else "❌ No"
        print(f"  [{i+1:>3}/{len(samples)}] {class_name:<15} {status:<12} {top3_names}")

        results.append({
            "index": i + 1,
            "class": class_name,
            "image": str(img_path),
            "top1_correct": top1,
            "top3_correct": top3,
            "identified": [ing["name"] for ing in ingredients],
            "error": None,
        })

        # Rate limit — free tier allows 15 RPM
        if i < len(samples) - 1:
            time.sleep(DELAY_SECONDS)

    # ── Compute overall metrics ───────────────────────────────────────────────
    valid     = [r for r in results if r["error"] is None]
    top1_acc  = sum(r["top1_correct"] for r in valid) / len(valid) if valid else 0
    top3_acc  = sum(r["top3_correct"] for r in valid) / len(valid) if valid else 0

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Images evaluated : {len(valid)}/{len(samples)}")
    print(f"  Top-1 Accuracy   : {top1_acc:.1%}  ({sum(r['top1_correct'] for r in valid)}/{len(valid)} correct)")
    print(f"  Top-3 Accuracy   : {top3_acc:.1%}  ({sum(r['top3_correct'] for r in valid)}/{len(valid)} correct)")

    print("\nPer-class breakdown:")
    print(f"  {'Class':<18} {'Top-1':>8} {'Top-3':>8} {'Total':>8}")
    print("  " + "-" * 46)
    for cls in sorted(per_class):
        d = per_class[cls]
        t1 = d["top1"] / d["total"] if d["total"] else 0
        t3 = d["top3"] / d["total"] if d["total"] else 0
        print(f"  {cls:<18} {t1:>7.1%} {t3:>8.1%} {d['total']:>8}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    output = {
        "config": {
            "dataset_path":     DATASET_PATH,
            "images_per_class": IMAGES_PER_CLASS,
            "random_seed":      RANDOM_SEED,
        },
        "overall": {
            "total_images":   len(samples),
            "evaluated":      len(valid),
            "top1_accuracy":  round(top1_acc, 4),
            "top3_accuracy":  round(top3_acc, 4),
        },
        "per_class": per_class,
        "per_image": results,
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n💾 Full results saved → evaluation_results.json")

    # CSV summary
    with open("evaluation_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "top1_correct", "top3_correct", "total",
                         "top1_accuracy", "top3_accuracy"])
        for cls in sorted(per_class):
            d  = per_class[cls]
            t1 = round(d["top1"] / d["total"], 4) if d["total"] else 0
            t3 = round(d["top3"] / d["total"], 4) if d["total"] else 0
            writer.writerow([cls, d["top1"], d["top3"], d["total"], t1, t3])
    print("💾 Summary CSV saved  → evaluation_summary.csv")
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()