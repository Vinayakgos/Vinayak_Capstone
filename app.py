"""
Recipe AI — Streamlit Interface
================================
Upload an ingredient image → get detected objects, identified ingredients,
and a full AI-generated recipe.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from PIL import Image

# Load API key from .env automatically — no need to paste it in the UI
load_dotenv(Path(__file__).parent.parent / ".env")
api_key = os.environ.get("GEMINI_API_KEY", "")

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import IngredientToRecipePipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rasoi (रसोई) AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f97316, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
}

.subtitle {
    color: #6b7280;
    font-size: 1.1rem;
    margin-top: 0.25rem;
}

.ingredient-chip {
    display: inline-block;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 999px;
    padding: 4px 14px;
    margin: 3px;
    font-size: 0.85rem;
    font-weight: 500;
}

.meta-pill {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #f97316, #ef4444);
    color: white;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 10px;
    flex-shrink: 0;
}

.nutrition-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
}

.nutrition-value { font-size: 1.6rem; font-weight: 700; color: #fb923c; }
.nutrition-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

.tip-box {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 6px 0;
    font-size: 0.9rem;
}

.detection-info {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #0369a1;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar — preferences only ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### Recipe Preferences")

    servings = st.slider("Servings", 1, 6, 2)

    dietary = st.multiselect(
        "Dietary preferences",
        ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free",
         "Low-Carb", "Halal"],
    )




# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Rasoi AI</div>', unsafe_allow_html=True)

st.markdown("---")


# ── API key guard — sourced from .env, not the UI ─────────────────────────────
if not api_key:
    st.error(
        " **GEMINI_API_KEY not found.** "
        "Make sure your `.env` file exists in the project root and contains:\n\n"
        "```\nGEMINI_API_KEY=AIza_your_key_here\n```\n\n"
        "Then restart the app with `streamlit run ui/app.py`."
    )
    st.stop()


# ── Image upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an ingredient image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Works best with clear, well-lit photos of ingredients laid out on a surface.",
)

run_btn = st.button(" Detect & Generate Recipe", type="primary", disabled=not uploaded)

if uploaded and not run_btn:
    st.image(uploaded, caption="Uploaded image — ready to analyse", use_container_width=True)


# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and uploaded:
    image = Image.open(uploaded).convert("RGB")

    status_box = st.empty()
    progress    = st.progress(0)

    def on_progress(msg):
        status_box.info(msg)
        stages = {"Stage 1": 0.33, "Stage 2": 0.66, "Stage 3": 0.90, "✅": 1.0}
        for k, v in stages.items():
            if k in msg:
                progress.progress(v)
                break

    try:
        pipeline = IngredientToRecipePipeline(
            api_key=api_key,
            yolo_model_size="n",
            yolo_conf=0.25,
        )

        result = pipeline.run(
            image_input=image,
            servings=servings,
            dietary_prefs=dietary,
            progress_callback=on_progress,
        )

        status_box.empty()
        progress.empty()

        detections  = result["detections"]
        annotated   = result["annotated_image"]
        vision_data = result["vision_data"]
        recipe      = result["recipe"]
        ingredients = vision_data["ingredients"]

        st.success(" Pipeline complete! Scroll down to see your recipe.")
        st.markdown("---")

        # ── Images (side by side — intentional, for comparison) ───────────
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)
        with img_col2:
            st.markdown("#### YOLOv8 Detections")
            st.image(annotated, use_container_width=True)
            if detections:
                det_labels = ", ".join(
                    f"{d['label']} ({d['confidence']:.0%})" for d in detections
                )
                st.markdown(
                    f'<div class="detection-info"> YOLO detected: {det_labels}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="detection-info"> YOLO found no standard COCO food objects — '
                    'Gemini Vision analysed the full image instead.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Identified ingredients ─────────────────────────────────────────
        st.markdown("###  Identified Ingredients")
        st.caption(f"_{vision_data.get('scene_description', '')}_")

        chips_html = "".join(
            f'<span class="ingredient-chip">'
            f'{"🟢" if i["confidence"]=="high" else "🟡" if i["confidence"]=="medium" else "🔴"} '
            f'{i["name"]} — {i["quantity_estimate"]}</span>'
            for i in ingredients
        )
        st.markdown(chips_html, unsafe_allow_html=True)

        with st.expander(" Full ingredient details (JSON)"):
            st.json(ingredients)

        st.markdown("---")

        # ── Recipe name + meta ─────────────────────────────────────────────
        st.markdown(f"###  {recipe['recipe_name']}")
        meta_html = (
            f'<span class="meta-pill"> {recipe["cuisine"]}</span>'
            f'<span class="meta-pill"> {recipe["difficulty"]}</span>'
            f'<span class="meta-pill"> Prep: {recipe["prep_time_minutes"]} min</span>'
            f'<span class="meta-pill"> Cook: {recipe["cook_time_minutes"]} min</span>'
            f'<span class="meta-pill"> Serves: {recipe["servings"]}</span>'
        )
        st.markdown(meta_html, unsafe_allow_html=True)
        st.markdown(f"\n_{recipe.get('description', '')}_")

        st.markdown("---")

        # ── Ingredients ────────────────────────────────────────────────────
        st.markdown("####  Ingredients")
        for ing in recipe.get("ingredients", []):
            note = f" _{ing['notes']}_" if ing.get("notes") else ""
            st.markdown(f"- **{ing['amount']} {ing['unit']}** {ing['item']}{note}")

        st.markdown("---")

        # ── Nutrition ──────────────────────────────────────────────────────
        # ── Nutrition ──────────────────────────────────────────────────────────────
        st.markdown("#### Nutrition per serving")
        n = recipe.get("nutrition_per_serving", {})
        st.markdown(
            f" **{n.get('calories', '—')} kcal** &nbsp;|&nbsp; "
            f" Protein: **{n.get('protein_g', '—')}g** &nbsp;|&nbsp; "
            f" Carbs: **{n.get('carbs_g', '—')}g** &nbsp;|&nbsp; "
            f" Fat: **{n.get('fat_g', '—')}g** &nbsp;|&nbsp; "
            f" Fibre: **{n.get('fibre_g', '—')}g**",
            unsafe_allow_html=True,
        )

        # ── Instructions ───────────────────────────────────────────────────
        st.markdown("#### Instructions")
        for step in recipe.get("instructions", []):
            time_txt = f" _{step.get('time_minutes', 0)} min_" if step.get("time_minutes") else ""
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;margin-bottom:14px">'
                f'<span class="step-number">{step["step"]}</span>'
                f'<div><strong>{step["title"]}</strong>{time_txt}<br>'
                f'<span style="color:#374151;font-size:0.9rem">{step["detail"]}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Chef's Tips ────────────────────────────────────────────────────
        tips = recipe.get("tips", [])
        if tips:
            st.markdown("#### Chef's Tips")
            for tip in tips:
                st.markdown(f'<div class="tip-box"> {tip}</div>', unsafe_allow_html=True)
            st.markdown("---")

        # ── Download ───────────────────────────────────────────────────────
        full_result_json = {
            "yolo_detections": [
                {"label": d["label"], "confidence": d["confidence"], "box": d["box"]}
                for d in detections
            ],
            "vision_data": vision_data,
            "recipe": recipe,
        }
        st.download_button(
            "Download full result (JSON)",
            data=json.dumps(full_result_json, indent=2),
            file_name="recipe_result.json",
            mime="application/json",
        )

    except Exception as e:
        status_box.empty()
        progress.empty()
        st.error(f"Pipeline error: {e}")
        with st.expander("Full traceback"):
            import traceback
            st.code(traceback.format_exc())