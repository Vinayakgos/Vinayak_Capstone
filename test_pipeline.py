"""
CLI test — run the full pipeline on a local image file.
Usage:
    python test_pipeline.py --image path/to/image.jpg --servings 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import IngredientToRecipePipeline

def main():
    parser = argparse.ArgumentParser(description="Ingredient → Recipe CLI")
    parser.add_argument("--image", required=True, help="Path to ingredient image")
    parser.add_argument("--servings", type=int, default=2)
    parser.add_argument("--dietary", nargs="*", default=[], help="e.g. Vegetarian Gluten-Free")
    parser.add_argument("--save", default="output.json", help="Save result JSON to file")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌  Set ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    pipeline = IngredientToRecipePipeline(api_key=api_key)

    result = pipeline.run(
        image_input=args.image,
        servings=args.servings,
        dietary_prefs=args.dietary,
        progress_callback=lambda msg: print(msg),
    )

    # Save annotated image
    result["annotated_image"].save("annotated_output.jpg")
    print("📸 Annotated image saved → annotated_output.jpg")

    # Print summary
    vision = result["vision_data"]
    recipe = result["recipe"]
    print(f"\n🥦 Ingredients found: {[i['name'] for i in vision['ingredients']]}")
    print(f"\n🍽️  Recipe: {recipe['recipe_name']}")
    print(f"   Cuisine: {recipe['cuisine']} | Difficulty: {recipe['difficulty']}")
    print(f"   Prep: {recipe['prep_time_minutes']} min | Cook: {recipe['cook_time_minutes']} min")
    print(f"\n📋 Instructions ({len(recipe['instructions'])} steps):")
    for step in recipe["instructions"]:
        print(f"   {step['step']}. {step['title']}: {step['detail'][:80]}…")

    # Save JSON
    save_data = {
        "yolo_detections": [
            {"label": d["label"], "confidence": d["confidence"], "box": d["box"]}
            for d in result["detections"]
        ],
        "vision_data": vision,
        "recipe": recipe,
    }
    with open(args.save, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n💾 Full result saved → {args.save}")


if __name__ == "__main__":
    main()