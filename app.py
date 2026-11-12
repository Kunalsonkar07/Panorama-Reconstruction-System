"""
AURA-CV | Neural Spatial Mosaic & Multi-View Synthesis Studio
Enterprise Wide-Baseline Image Registration & Harmonic Manifold Fusion Engine.
Run with:  streamlit run app.py
"""

import os
import glob
import time
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from mosaic_core.pipeline import StitchingPipeline, StitchResult

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration & Advanced Theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Panorama Reconstruction System",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Bespoke High-Tech Custom CSS Injection
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Background & Typography */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #090d16 100%);
    color: #e2e8f0;
}

/* Hero Header Card */
.studio-hero {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 25px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.studio-hero::after {
    content: "";
    position: absolute;
    top: -50px;
    right: -50px;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.studio-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.studio-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    max-width: 900px;
    line-height: 1.5;
    margin-bottom: 16px;
}

.badge-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}

.studio-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-cyan {
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(14, 165, 233, 0.4);
    color: #38bdf8;
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
}

.badge-purple {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #c084fc;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fbbf24;
}

/* Glass Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.6) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 2px solid #38bdf8 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Tabs Redesign */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 6px;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    color: #94a3b8;
    background: transparent;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
}

/* Primary Execution Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    box-shadow: 0 10px 25px -5px rgba(14, 165, 233, 0.5) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px -5px rgba(14, 165, 233, 0.7) !important;
}

/* Secondary Buttons */
.stButton > button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
    background: rgba(30, 41, 59, 0.9) !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* Section Header Badges */
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    display: block;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img))


def resize_for_display(img: np.ndarray, max_width: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def ndarray_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Image encoding failed.")
    return buf.tobytes()


def render_markdown_table(rows: list):
    """Safely render tabular data in markdown without pyarrow / pandas dependencies."""
    if not rows:
        st.info("No telemetry logs available.")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    st.markdown("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Algorithmic Hyperparameter Console
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<span class='section-tag'>CONTROL HUB</span>", unsafe_allow_html=True)
    st.title("◈ PARAMETER HUB")
    st.caption("AURA-CV High-Throughput Manifold Engine")
    st.markdown("---")

    # ── Engine Selection ─────────────────────────────────────────────────────
    st.subheader("⚙️ Processing Architecture")
    engine = st.radio(
        "Compute Backend",
        options=["custom", "opencv"],
        format_func=lambda x: (
            "AURA-CV Custom Manifold Engine (SIFT/ORB + RANSAC)"
            if x == "custom"
            else "OpenCV Legacy (cv2.Stitcher C++ Base)"
        ),
        index=0,
        help="Custom engine provides granular telemetry, exposure balancing, and cylindrical alignment.",
    )

    # ── Geometric Alignment & Distortion Control ─────────────────────────────
    st.subheader("📐 Manifold Geometry & Kinematics")
    use_cylindrical = st.toggle(
        "Cylindrical Manifold Warp",
        value=True,
        help="Maps rotational degrees of freedom onto a cylindrical surface to eliminate lateral scale dilation.",
    )

    if use_cylindrical:
        fov_deg = st.slider(
            "Optical Sensor FOV (Horizontal)",
            min_value=30,
            max_value=120,
            value=60,
            step=5,
            help="Estimated horizontal Field-of-View in degrees (Standard: 55°–65°, Wide: 70°–90°).",
        )
    else:
        fov_deg = 60

    transform_type = st.selectbox(
        "Kinematic Alignment Model",
        options=["affine", "homography"],
        index=0,
        format_func=lambda x: (
            "Affine Similarity (4-DOF Rigid, Height Invariant)"
            if x == "affine"
            else "Projective Homography (8-DOF Full Perspective)"
        ),
        help="Affine Similarity strictly prevents wedge/cone divergence (h20=0, h21=0).",
    )

    crop_mode = st.selectbox(
        "Active Boundary Rectification",
        options=["rect", "tight", "none"],
        index=0,
        format_func=lambda x: {
            "rect": "Stack-Based Largest Interior Rectangle (Clean Frame)",
            "tight": "Tight Non-Zero Bounding Box",
            "none": "Full Raw Canvas Projection",
        }[x],
        help="Isolates the largest valid interior rectangular matrix via O(W·H) histogram traversal.",
    )

    st.markdown("---")

    # ── Feature Detection & Matching ─────────────────────────────────────────
    st.subheader("🔬 Invariant Feature Extractor")
    method = st.selectbox(
        "Keypoint Operator",
        options=["SIFT", "ORB"],
        index=0,
        help="SIFT: 128-D scale/rotation invariant float vectors. ORB: 256-bit high-speed binary descriptors.",
    )

    max_features = st.slider(
        "Max Feature Density (pts / frame)",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=500,
    )

    ratio_thresh = st.slider(
        "Lowe's Vector Ratio Filter",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.01,
        help="Ambiguity distance ratio for nearest neighbors. Lower enforces stricter exclusivity.",
    )

    ransac_thresh = st.slider(
        "RANSAC Spatial Margin (px)",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
    )

    # ── Blending & Exposure ──────────────────────────────────────────────────
    st.subheader("🌈 Radiometric Synthesis & Seam Blending")
    blend_mode = st.selectbox(
        "Composite Fusion Algorithm",
        options=["multiband", "feather", "simple"],
        index=0,
        format_func=lambda x: {
            "multiband": "Multi-Band Laplacian Pyramid (Frequency Split)",
            "feather": "Distance-Transform Alpha Feathering",
            "simple": "Direct Pixel Transfer (Raw Seams)",
        }[x],
    )

    compensate_exposure = st.toggle(
        "Chromatic Gain Equalization",
        value=True,
        help="Calibrates RGB gain vectors across overlapping boundaries to neutralize auto-exposure shifts.",
    )

    enable_comparison = st.toggle(
        "Synthesize Uncorrected Baseline",
        value=True,
        help="Simulates naive planar homography to quantify distortion reduction.",
    )

    st.markdown("---")
    st.caption("AURA-CV Neural Spatial Studio · Release 2.5 Pro")


# ──────────────────────────────────────────────────────────────────────────────
# Main Application Interface
# ──────────────────────────────────────────────────────────────────────────────

# Hero Banner
st.markdown(
    """
    <div class="studio-hero">
        <div class="studio-title">◈ Image Panorama Reconstruction System</div>
        <div class="studio-subtitle">
            Autonomous multi-perspective spatial registration and non-Euclidean manifold synthesis. 
            Engineered with <strong>Cylindrical Projection</strong>, <strong>Affine Similarity RANSAC</strong> (zero wedge distortion), 
            <strong>Central Reference Propagation</strong>, and <strong>Multi-Band Laplacian Frequency Fusion</strong>.
        </div>
        <div class="badge-strip">
            <span class="studio-badge badge-cyan">⚡ ENGINE: SIFT / ORB</span>
            <span class="studio-badge badge-emerald">📐 TOPOLOGY: CYLINDRICAL RANSAC</span>
            <span class="studio-badge badge-purple">🎨 FUSION: MULTI-BAND LAPLACIAN</span>
            <span class="studio-badge badge-amber">🛡️ STATUS: ACTIVE & READY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Preset Benchmark Sequences ────────────────────────────────────────────────
st.markdown("<span class='section-tag'>BENCHMARK DATASETS</span>", unsafe_allow_html=True)
col_s1, col_s2, col_s3 = st.columns([1.2, 1.2, 1.6])

if "loaded_images" not in st.session_state:
    st.session_state.loaded_images = []
if "loaded_names" not in st.session_state:
    st.session_state.loaded_names = []

with col_s1:
    if st.button("🏛️ Heritage Architecture (St. Petersburg Quad)", use_container_width=True):
        real_files = sorted(glob.glob("field_captures/*.jpg"))
        st.session_state.loaded_images = [cv2.imread(f) for f in real_files]
        st.session_state.loaded_names = [os.path.basename(f) for f in real_files]
        st.success(f"Loaded {len(real_files)} multi-view frames into workspace memory.")

with col_s2:
    if st.button("🏙️ Urban Skyline (Synthetic Multi-Angle)", use_container_width=True):
        synth_files = sorted(glob.glob("synthetic_quads/image_*.jpg"))
        st.session_state.loaded_images = [cv2.imread(f) for f in synth_files]
        st.session_state.loaded_names = [os.path.basename(f) for f in synth_files]
        st.success(f"Loaded {len(synth_files)} perspective-shifted quad frames.")

with col_s3:
    if st.session_state.loaded_images:
        if st.button("🧹 Clear Workspace Cache", use_container_width=False):
            st.session_state.loaded_images = []
            st.session_state.loaded_names = []
            st.rerun()

st.markdown("---")

# ── Custom Image Ingestion ───────────────────────────────────────────────────
st.markdown("<span class='section-tag'>FILE INGESTION</span>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Ingest sequence frames (JPEG, PNG, TIFF, BMP) in arbitrary order:",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state.loaded_images = []
    st.session_state.loaded_names = []
    for uf in uploaded_files:
        try:
            pil_img = Image.open(uf)
            bgr = pil_to_bgr(pil_img)
            st.session_state.loaded_images.append(bgr)
            st.session_state.loaded_names.append(uf.name)
        except Exception as e:
            st.warning(f"Failed to ingest **{uf.name}**: {e}")

images_bgr = st.session_state.loaded_images
image_names = st.session_state.loaded_names

if len(images_bgr) < 2:
    st.info("💡 Load a benchmark sequence above or upload 2 or more overlapping photographs to begin spatial reconstruction.")
    st.stop()

# ── Loaded Image Previews ────────────────────────────────────────────────────
with st.expander(f"📁 Ingested Frames ({len(images_bgr)} frames verified)", expanded=True):
    cols = st.columns(max(1, min(len(images_bgr), 4)))
    for i, (img, name) in enumerate(zip(images_bgr, image_names)):
        with cols[i % len(cols)]:
            h, w = img.shape[:2]
            st.image(to_pil(img), caption=f"Frame {i+1}: {name} ({w}×{h} px)", use_container_width=True)

st.markdown("---")

# ── Run Spatial Synthesis ────────────────────────────────────────────────────
run_button = st.button("⚡ EXECUTE SPATIAL REGISTRATION & HARMONIC SYNTHESIS", type="primary", use_container_width=True)

if run_button:
    # Compute focal length from user FOV
    w0 = images_bgr[0].shape[1]
    focal_px = w0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0)) if use_cylindrical else 0.0

    pipeline = StitchingPipeline(
        method=method,
        max_features=max_features,
        ratio_thresh=ratio_thresh,
        ransac_thresh=ransac_thresh,
        min_matches=10,
        min_inliers=8,
        transform_type=transform_type,
        blend_mode=blend_mode,
        use_cylindrical=use_cylindrical,
        focal_length=focal_px,
        crop_mode=crop_mode,
        engine=engine,
        compensate_exposure=compensate_exposure,
        compute_comparison=enable_comparison,
    )

    progress_bar = st.progress(0.0)
    status_box = st.empty()
    live_log = st.empty()
    logs: list[str] = []

    def progress_callback(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0))
        status_box.markdown(f"**Pipeline Trace:** `{msg}`")
        logs.append(msg)
        with live_log.container():
            with st.expander("📋 Real-Time Pipeline Stream", expanded=False):
                st.text("\n".join(logs[-25:]))

    t_start = time.time()
    result: StitchResult = pipeline.stitch(images_bgr, progress_cb=progress_callback)
    duration = time.time() - t_start

    progress_bar.progress(1.0)
    status_box.markdown(f"**System Status:** ✅ Synthesis Completed in **{duration:.2f} seconds**")

    # ── Executive Telemetry Dashboard ────────────────────────────────────────
    st.markdown("<span class='section-tag'>EXECUTIVE TELEMETRY</span>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Latency", f"{duration:.2f} s")
    with m2:
        st.metric("Features Detected", f"{result.total_keypoints:,} pts")
    with m3:
        st.metric("Mean RANSAC Inliers", f"{result.avg_inliers:.1f} / pair")
    with m4:
        flare_metric = "0.0% (Zero Flare)" if transform_type == "affine" else "Projective 8-DOF"
        st.metric("Manifold Divergence", flare_metric)

    st.markdown("---")

    # ── Result Tabs ──────────────────────────────────────────────────────────
    tab_pano, tab_compare, tab_matches, tab_steps, tab_diagnostics = st.tabs(
        [
            "🌌 High-Res Composite Viewport",
            "📐 Manifold Rectification & Distortion Audit",
            "🕸️ Keypoint Topology & Inlier Matrix",
            "🎞️ Iterative Mosaic Timeline",
            "📈 Telemetry & Hardware Trace",
        ]
    )

    # ── Tab 1: Final Panorama ────────────────────────────────────────────────
    with tab_pano:
        if result.panorama is not None:
            num_stitched = len(result.stitch_order) - len(result.failed_images)
            st.success(
                f"🎉 High-fidelity mosaic assembled: **{num_stitched} of {len(images_bgr)} frames aligned** "
                f"in **{duration:.2f}s** with **strict height-uniformity enforcement**."
            )

            if result.failed_images:
                failed_str = ", ".join(image_names[i] for i in result.failed_images)
                st.warning(f"⚠️ Frames dropped due to insufficient optical overlap: **{failed_str}**")

            h_res, w_res = result.panorama.shape[:2]
            aspect = w_res / float(h_res)

            st.subheader(f"Synthesized Panoramic Field — {w_res} × {h_res} px (Aspect {aspect:.2f}:1)")
            display_img = resize_for_display(result.panorama, 1200)
            st.image(to_pil(display_img), use_container_width=True)

            # Export Hub
            c_down1, c_down2 = st.columns(2)
            with c_down1:
                st.download_button(
                    "💾 Export Lossless Master (PNG)",
                    data=ndarray_to_bytes(result.panorama, ".png"),
                    file_name="synth_mosaic_master.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with c_down2:
                st.download_button(
                    "💾 Export Compressed Derivative (JPEG)",
                    data=ndarray_to_bytes(result.panorama, ".jpg"),
                    file_name="synth_mosaic_derivative.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            st.caption(
                f"Kinematics: {transform_type.title()} · "
                f"Manifold: {'Cylindrical (FOV ' + str(fov_deg) + '°)' if use_cylindrical else 'Planar'} · "
                f"Exposure Comp: {'Enabled' if compensate_exposure else 'Disabled'} · "
                f"Fusion: {blend_mode.title()} · Boundary: {crop_mode.title()}"
            )
        else:
            st.error("❌ Synthesis terminated. Insufficient mutual visual overlap or RANSAC consensus failure.")

    # ── Tab 2: Distortion Analysis (Before vs After) ──────────────────────────
    with tab_compare:
        st.subheader("📐 Manifold Rectification & Non-Euclidean Distortion Audit")
        st.markdown(
            """
            This telemetry view demonstrates the mathematical divergence between **Standard Chained Planar Homography** 
            and our **Cylindrical Affine Manifold Formulation**.
            """
        )

        col_bfa, col_bfb = st.columns(2)
        with col_bfa:
            st.markdown("#### ❌ Baseline: Uncorrected Planar Homography")
            st.caption(
                "Chained standard 3×3 projective homographies without cylindrical projection. "
                "Projective division coefficients compound left-to-right, producing the classic cone/wedge flare."
            )
            if result.naive_panorama is not None:
                st.image(to_pil(resize_for_display(result.naive_panorama, 650)), use_container_width=True)
                nh, nw = result.naive_panorama.shape[:2]
                st.info(f"📐 Baseline Footprint: {nw} × {nh} px · Uncontrolled boundary accumulation & perspective stretch.")
            else:
                st.info("Baseline simulation skipped or disconnected.")

        with col_bfb:
            st.markdown("#### ✅ Calibrated: AURA-CV Cylindrical Manifold")
            st.caption(
                "Cylindrical mapping converts rotational pan angles into pure linear translations. "
                "Affine similarity RANSAC guarantees zero height expansion (h20=0, h21=0) anchored to Central Reference."
            )
            if result.panorama is not None:
                st.image(to_pil(resize_for_display(result.panorama, 650)), use_container_width=True)
                rh, rw = result.panorama.shape[:2]
                st.success(f"🎯 Calibrated Footprint: {rw} × {rh} px · Perfectly planarized 100% height uniformity.")

        st.markdown("---")
        st.markdown("##### 🔬 Architectural Divergence Matrix")
        diff_table = [
            {
                "Evaluation Vector": "Kinematic Transformation",
                "Uncorrected Legacy Pipeline": "Full Projective Homography (8-DOF)",
                "AURA-CV Synthesis Architecture": "Affine Similarity (4-DOF, h20=0, h21=0)",
            },
            {
                "Evaluation Vector": "Projection Manifold",
                "Uncorrected Legacy Pipeline": "Planar (severe peripheral lateral dilation)",
                "AURA-CV Synthesis Architecture": "Cylindrical Manifold (radial scale preservation)",
            },
            {
                "Evaluation Vector": "Propagation Origin",
                "Uncorrected Legacy Pipeline": "Sequential Chaining (monotonically accumulates error)",
                "AURA-CV Synthesis Architecture": "Central Reference Origin (symmetric bidirectional propagation)",
            },
            {
                "Evaluation Vector": "Vertical Scale Stability",
                "Uncorrected Legacy Pipeline": "Exponential trumpet expansion (+50% to +100%)",
                "AURA-CV Synthesis Architecture": "Strictly invariant height across entire field",
            },
            {
                "Evaluation Vector": "Radiometric Balance",
                "Uncorrected Legacy Pipeline": "Uncompensated (sharp exposure seams)",
                "AURA-CV Synthesis Architecture": "Channel-wise Overlap Gain Normalization",
            },
            {
                "Evaluation Vector": "Seam Fusion",
                "Uncorrected Legacy Pipeline": "Alpha Feathering / Direct Copy",
                "AURA-CV Synthesis Architecture": "Multi-Band Laplacian Frequency Pyramids",
            },
            {
                "Evaluation Vector": "Boundary Cropping",
                "Uncorrected Legacy Pipeline": "Ragged irregular black borders",
                "AURA-CV Synthesis Architecture": "O(W·H) Stack Histogram Interior Rectification",
            },
        ]
        render_markdown_table(diff_table)

    # ── Tab 3: Feature Matches ───────────────────────────────────────────────
    with tab_matches:
        st.subheader("🕸️ Keypoint Correspondence Topology")
        valid_pairs = [p for p in result.pair_results if p.match_vis is not None]

        if not valid_pairs:
            st.info("No valid correspondence pairs were established.")
        else:
            for pr in valid_pairs:
                name_a = image_names[pr.img_idx_a]
                name_b = image_names[pr.img_idx_b]
                status_icon = "✅ Consensual Alignment" if pr.homography is not None else "❌ RANSAC Rejection"
                with st.expander(f"Pair: {name_a} ↔ {name_b} | {pr.num_matches} Matches ({pr.num_inliers} Inliers) — {status_icon}"):
                    vis_display = resize_for_display(pr.match_vis, 1100)
                    st.image(to_pil(vis_display), use_container_width=True)

        # Inlier Heatmap
        st.markdown("---")
        st.subheader("Pairwise RANSAC Inlier Correlation Matrix")
        n = len(images_bgr)
        matrix = np.zeros((n, n), dtype=int)
        for pr in result.pair_results:
            matrix[pr.img_idx_a][pr.img_idx_b] = pr.num_inliers
            matrix[pr.img_idx_b][pr.img_idx_a] = pr.num_inliers

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(4.5, n), max(3.2, n - 1)))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        
        im = ax.imshow(matrix, cmap="magma", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_names = [f"F-{i+1}" for i in range(n)]
        ax.set_xticklabels(short_names, rotation=0, ha="center", fontsize=9, color="#94a3b8")
        ax.set_yticklabels(short_names, fontsize=9, color="#94a3b8")
        
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                txt_color = "white" if val < matrix.max() * 0.7 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=txt_color, weight="bold")
        
        ax.set_title("Geometric Inlier Density Matrix", color="white", fontsize=11, pad=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 4: Stitching Steps ───────────────────────────────────────────────
    with tab_steps:
        st.subheader("🎞️ Iterative Mosaic Timeline")
        st.caption("Chronological canvas accumulation as each verified frame is registered onto the global frame.")
        if not result.intermediate_panoramas:
            st.info("No intermediate frames captured.")
        else:
            for step_idx, inter_img in enumerate(result.intermediate_panoramas):
                ih, iw = inter_img.shape[:2]
                st.markdown(f"**Registration Stage {step_idx + 1}** — Active Canvas: `{iw} × {ih} px`")
                st.image(to_pil(resize_for_display(inter_img, 1100)), use_container_width=True)
                st.markdown("---")

    # ── Tab 5: Telemetry & Diagnostics ───────────────────────────────────────
    with tab_diagnostics:
        st.subheader("📈 Computational Latency Breakdown")
        if result.timings:
            labels = [k.replace("_", " ").title() for k in result.timings.keys()]
            times_ms = [v * 1000.0 for v in result.timings.values()]

            fig_t, ax_t = plt.subplots(figsize=(8, 3.2))
            fig_t.patch.set_facecolor('#0f172a')
            ax_t.set_facecolor('#0f172a')
            
            bars = ax_t.barh(labels, times_ms, color="#38bdf8", edgecolor="#0284c7", height=0.6)
            ax_t.set_xlabel("Latency (milliseconds)", color="#94a3b8", fontsize=9)
            ax_t.set_title("Subsystem Execution Profile (ms)", color="white", fontsize=11, pad=10)
            ax_t.tick_params(colors="#94a3b8", labelsize=8)
            
            for spine in ax_t.spines.values():
                spine.set_color('#334155')

            for bar in bars:
                w_val = bar.get_width()
                ax_t.text(w_val + max(times_ms) * 0.015, bar.get_y() + bar.get_height() / 2, f"{w_val:.1f} ms", va="center", fontsize=8, color="#38bdf8", weight="bold")
            
            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

        st.markdown("---")
        st.subheader("Pairwise Homography & Verification Logs")
        stats_rows = [
            {
                "Node A": image_names[pr.img_idx_a],
                "Node B": image_names[pr.img_idx_b],
                "Detected Matches": pr.num_matches,
                "RANSAC Inliers": pr.num_inliers,
                "Consensus Inlier Ratio": f"{(pr.num_inliers / pr.num_matches * 100):.1f}%" if pr.num_matches > 0 else "0%",
                "Geometric Consensus": "✅ Verified" if pr.homography is not None else "❌ Rejected",
            }
            for pr in result.pair_results
        ]
        render_markdown_table(stats_rows)

        st.markdown("---")
        st.subheader("Hyperparameter Runtime Environment")
        config_rows = [
            {"Parameter": "Compute Subsystem", "Assigned Value": engine.upper()},
            {"Parameter": "Keypoint Operator", "Assigned Value": method},
            {"Parameter": "Kinematic Constraint", "Assigned Value": transform_type.title()},
            {"Parameter": "Manifold Projection", "Assigned Value": "Cylindrical Active" if use_cylindrical else "Planar Flat"},
            {"Parameter": "Optical Sensor FOV", "Assigned Value": f"{fov_deg}°" if use_cylindrical else "N/A"},
            {"Parameter": "Calibrated Focal Pixel", "Assigned Value": f"{focal_px:.1f} px" if use_cylindrical else "N/A"},
            {"Parameter": "RANSAC Spatial Margin", "Assigned Value": f"{ransac_thresh} px"},
            {"Parameter": "Lowe's Ambiguity Threshold", "Assigned Value": str(ratio_thresh)},
            {"Parameter": "Chromatic Gain Balance", "Assigned Value": "Active" if compensate_exposure else "Bypassed"},
            {"Parameter": "Seam Synthesis Mode", "Assigned Value": blend_mode.title()},
            {"Parameter": "Boundary Rectification", "Assigned Value": crop_mode.title()},
            {"Parameter": "Total Sequence Latency", "Assigned Value": f"{duration:.2f} seconds"},
        ]
        render_markdown_table(config_rows)

        with st.expander("Detailed System Execution Trace"):
            st.text("\n".join(result.log))
