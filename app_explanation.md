# 🔭 Panorama Stitching & Reconstruction System — Full Technical Explanation

## What is This Application?

This is a **Panorama Image Stitching System** — a web-based tool that takes multiple overlapping photographs (taken by rotating a camera from left to right) and automatically stitches them into a single wide, seamless, high-resolution panoramic image.

It is built with:
- **Python** as the core language
- **OpenCV (`cv2`)** for all computer vision operations
- **NumPy** for matrix mathematics
- **Streamlit** as the interactive web UI
- **Pillow (PIL)** and **Matplotlib** for image display and visualization

---

## 🗺️ Full Pipeline Flow — Step by Step

```mermaid
flowchart TD
    A["📂 Input Images\n(Upload or Load Samples)"] --> B["Step 1: Cylindrical Projection\ncylindrical.py"]
    B --> C["Step 2: Feature Detection\nSIFT or ORB — feature_detector.py"]
    C --> D["Step 3: Pairwise Feature Matching\nFLANN / BFMatcher + Lowe's Ratio Test\nfeature_matcher.py"]
    D --> E["Step 4: Graph-Based Image Ordering\nGreedy inlier count graph — ordering.py"]
    E --> F["Step 5: RANSAC Transformation Estimation\nAffine or Homography — homography.py"]
    F --> G["Step 6: Central Reference Frame\nAll images aligned to middle image\npipeline.py"]
    G --> H["Step 7: Warp to Global Canvas\nwarper.py"]
    H --> I["Step 8: Seamless Blending\nMulti-band Laplacian / Feather / Simple\nwarper.py"]
    I --> J["Step 9: Largest Rectangle Crop\nO(W×H) histogram — rect_crop.py"]
    J --> K["🖼️ Final Panorama\nUniform Height · No Black Borders · Seamless"]
```

---

## Step 1 — Cylindrical Projection (`cylindrical.py`)

> [!IMPORTANT]
> This step prevents the most common failure of naive stitching: **wedge / trapezoid distortion** where the right side of the panorama flares outward like a horn or cone.

### What it does
When a camera rotates around its vertical axis to capture a scene, each photograph represents a different **angular slice**. Standard flat (planar) stitching maps these rotations using projective homographies, but chaining multiple homographies causes the perspective division terms ($h_{20}, h_{21}$) to compound and blow up — producing a cone-shaped "trumpet" distortion.

**Cylindrical projection** solves this by mathematically "unrolling" the camera rotation onto the surface of an imaginary cylinder. After projection, what was a **rotation** becomes a **pure horizontal translation**, which the Affine transform handles perfectly.

### The Math (Inverse Mapping)
For each output pixel $(x_{cyl}, y_{cyl})$ on the cylinder, we compute where it came from in the original flat photo:

$$\theta = \frac{x_{cyl} - c_x}{f} \quad \Rightarrow \quad x_{src} = f \cdot \tan(\theta) + c_x$$

$$y_{src} = \frac{y_{cyl} - c_y}{\cos(\theta)} + c_y$$

Where $f$ = focal length in pixels, $c_x, c_y$ = image center. The code builds lookup maps (`map_x`, `map_y`) and uses `cv2.remap()` for fast bilinear resampling.

### Focal Length Estimation
Estimated from the assumed horizontal Field-of-View (default 60°):

$$f = \frac{W}{2 \cdot \tan(\text{FOV}/2)}$$

After warping, `strip_black_cols()` removes the black curved edges that appear at the left/right borders.

---

## Step 2 — Feature Detection (`feature_detector.py`)

The app supports **two detectors**, selectable from the sidebar:

### SIFT — Scale-Invariant Feature Transform (Default & Recommended)

SIFT works in four stages:

1. **Scale-Space Extrema Detection**: Builds a Gaussian pyramid (blurred images at multiple scales), then computes Difference-of-Gaussians (DoG) between adjacent levels. Local maxima/minima in this 3D scale-space are candidate keypoints.

2. **Keypoint Localization**: Fits a quadratic to each candidate, refining position and scale to sub-pixel accuracy. Rejects low-contrast points and edge responses (using the Harris criterion).

3. **Orientation Assignment**: Computes gradient directions in the keypoint's local patch. Assigns the dominant orientation → makes the descriptor **rotation-invariant**.

4. **Descriptor Construction**: Samples a 16×16 patch (at the keypoint's scale), divides into 4×4 sub-blocks, computes an 8-bin gradient histogram per block → **128-dimensional float vector**.

**Why SIFT is powerful**: The 128-D descriptor is highly distinctive and invariant to scale, rotation, and moderate illumination change. It reliably finds the same physical corner/texture in photos taken from different angles.

### ORB — Oriented FAST and Rotated BRIEF (Alternative)

ORB is a fast, patent-free alternative to SIFT:

1. **FAST Corner Detector**: Checks 16 pixels on a circle around each candidate. If N consecutive pixels are all brighter or all darker than the center by a threshold, it is a corner.

2. **Harris Score**: Ranks corners by their Harris response to retain only the strongest.

3. **Orientation**: Uses image moments in a patch to assign an orientation → rotation-invariant.

4. **BRIEF Descriptor**: Compares the intensity of random pixel pairs in a patch, encoding each comparison as a single bit → compact **256-bit binary string** (32 bytes).

**Why ORB is useful**: Much faster than SIFT, good for real-time or mobile applications. Less accurate for wide-angle stitching.

```
Feature Detector Comparison:
┌─────────────────┬────────────────┬─────────────────┐
│ Property        │ SIFT           │ ORB             │
├─────────────────┼────────────────┼─────────────────┤
│ Descriptor Type │ Float (128-D)  │ Binary (256-bit)│
│ Scale Invariant │ ✅ Yes         │ ✅ Yes          │
│ Rotation Inv.   │ ✅ Yes         │ ✅ Yes          │
│ Speed           │ Slower         │ Fast            │
│ Accuracy        │ High           │ Moderate        │
│ Best Used For   │ Quality panos  │ Fast / real-time│
└─────────────────┴────────────────┴─────────────────┘
```

---

## Step 3 — Feature Matching (`feature_matcher.py`)

After every image has keypoints + descriptors, we find which keypoints in Image A correspond to the same physical point in Image B.

### Two Matchers (chosen automatically)

**FLANN (Fast Library for Approximate Nearest Neighbors)** — used with SIFT:
- Builds a **KD-tree** index over the 128-D SIFT float descriptors
- For each descriptor in Image A, finds its **2 nearest neighbors** in Image B
- Extremely fast for large float descriptor sets

**BFMatcher (Brute-Force Matcher)** — used with ORB:
- Compares every ORB descriptor in Image A against every descriptor in Image B using **Hamming distance** (counts differing bits)
- More exhaustive but correct for small binary descriptors

### Lowe's Ratio Test
Raw matches contain many false positives. For each match, we find the **best match** ($d_1$) and **second-best match** ($d_2$) and keep the match only if:

$$\frac{d_1}{d_2} < 0.75$$

**Why**: A genuine match has a much smaller distance to its true correspondence than to any other descriptor. If the ratio is high (≈1.0), the match is likely ambiguous (e.g., a repetitive texture wall) and is discarded.

---

## Step 4 — Graph-Based Image Ordering (`ordering.py`)

> [!NOTE]
> When a user uploads images in random order, the pipeline must automatically discover the correct left-to-right sequence before stitching.

After all pairs are matched, we have an inlier-count matrix used as a weighted graph:
- **Nodes** = individual images
- **Edge weights** = number of RANSAC inliers between each pair

### Greedy Chain Algorithm
1. **Seed**: Find the pair with the **highest inlier count** (most overlap) → start the chain.
2. **Grow Right**: From the rightmost image, find the unvisited image with the most matches → append.
3. **Grow Left**: Check if any unvisited image matches better to the leftmost → prepend.
4. Repeat until all images are placed.

This discovers the correct ordering regardless of upload order.

---

## Step 5 — Transformation Estimation with RANSAC (`homography.py`)

For each adjacent pair in the ordered sequence, we compute the geometric transformation mapping one image's coordinate space to the other's.

### Affine / Similarity Transform (Default)
A **4-DOF similarity transform**: translation + rotation + uniform scale.

$$H = \begin{pmatrix} s\cos\theta & -s\sin\theta & t_x \\ s\sin\theta & s\cos\theta & t_y \\ 0 & 0 & 1 \end{pmatrix}$$

The key insight: the **bottom row is always [0, 0, 1]** — no perspective division ever occurs. This guarantees that parallel horizontal lines stay horizontal and image heights remain perfectly uniform across the whole panorama.

### Full Homography (Optional)
An **8-DOF projective transform** (full 3×3 matrix). Can represent any planar mapping, but the bottom-row terms $h_{20}, h_{21}$ cause perspective shrink/stretch — leading to wedge distortion when multiple homographies are chained.

### RANSAC — Random Sample Consensus
RANSAC finds the correct transform even with many false matches (outliers):

1. **Randomly sample** the minimum matches needed (4 for homography, 3 for affine).
2. **Fit** a candidate transformation to that tiny sample.
3. **Count inliers**: how many of all matches agree (reprojection error < 4px threshold).
4. **Repeat** up to 2000 iterations.
5. **Keep** the transformation with the **most inliers** and refit using all of them.

Result: reliable estimation even when 50–70% of matches are wrong.

---

## Step 6 — Central Reference Frame (`pipeline.py`)

> [!TIP]
> Instead of chaining transforms from left to right (which accumulates blur and drift), ALL images are aligned directly to the **middle image** of the sequence.

For 5 images [A, B, C, D, E] with C as reference:
- A → C: $H_{B \rightarrow A}^{-1} \cdot H_{C \rightarrow B}^{-1}$
- B → C: $H_{C \rightarrow B}^{-1}$
- C → C: Identity
- D → C: $H_{D \rightarrow C}$
- E → C: $H_{D \rightarrow C} \cdot H_{E \rightarrow D}$

The maximum chain length is halved ($N/2$ instead of $N-1$), symmetrically minimizing cumulative error from both sides. No image undergoes more than $N/2$ transformation compositions, so blur and drift are minimized.

---

## Steps 7 & 8 — Warping and Blending (`warper.py`)

### Warping to Global Canvas
1. Each image's 4 corners are transformed by its $H_{to\_ref}$ matrix (`cv2.perspectiveTransform()`) to find where it lands in reference coordinates.
2. The bounding box of all transformed corners defines the **global canvas** size.
3. A translation matrix $T$ shifts coordinates so the canvas top-left is at origin (0,0).
4. Each image is warped onto the canvas: `cv2.warpPerspective(img, T @ H_to_ref, canvas_size)`.

### Three Blending Modes

**1. Multi-band Laplacian Pyramid Blending (Default)**
The most sophisticated blending — blends low-frequency (color/illumination) and high-frequency (edges/texture) content separately:

1. Build a **Gaussian pyramid** (blurred/downsampled versions) for each image and the alpha weight mask.
2. Build a **Laplacian pyramid**: $L_k = G_k - \text{upsample}(G_{k+1})$. Each level captures details at that frequency band.
3. **Blend each pyramid level** separately using the alpha mask (smooth, from distance transform).
4. **Reconstruct** by summing back up the pyramid.

Result: **no visible seams**, even with illumination differences, because low-frequency color is blended over a wide boundary while sharp edges are preserved over a narrow one.

**2. Feather / Distance-Transform Blending**
Each pixel in the overlap zone gets a weight equal to its **distance to the nearest image boundary**:

$$\alpha_A(x,y) = \frac{d_A(x,y)}{d_A(x,y) + d_B(x,y)}$$

Produces smooth, gradual cross-dissolves across overlapping regions.

**3. Simple Overlay**
No blending — each new image just overwrites the previous one. Fast but produces visible seams at boundaries.

---

## Step 9 — Automatic Rectangle Crop (`rect_crop.py`)

After blending, the canvas has black (zero) regions wherever no image contributed — irregular curved borders from cylindrical warping and slight rotations.

### O(W×H) Histogram Algorithm
1. For each **row** of the binary validity mask, build a **column-height histogram**: for each column, how many consecutive valid rows exist above?
2. Apply the **Largest Rectangle in Histogram** (classic stack-based O(n) algorithm) to find the widest rectangle fitting in that row.
3. Track the globally largest such rectangle across all rows.

This finds the **largest possible axis-aligned rectangle** contained entirely within valid pixels — guaranteeing no black edges or borders in the final output.

---

## Streamlit UI Features (`app.py`)

| Feature | Description |
|:--------|:------------|
| **Sample Loaders** | 1-click load of real St. Petersburg river photos or synthetic cityscape |
| **File Uploader** | Drag-and-drop any overlapping JPG/PNG photos |
| **Sidebar Controls** | Feature detector (SIFT/ORB), max features, ratio threshold, transform type, blend mode, cylindrical toggle, crop mode, OpenCV engine toggle |
| **Live Progress Bar** | Step-by-step progress with percentage and status messages |
| **Feature Match Viewer** | Side-by-side images with colored lines showing matched keypoints |
| **Pairwise Inlier Heatmap** | Matrix heatmap of RANSAC inliers between each image pair |
| **Intermediate Steps** | Animated step-by-step view of images being added to the panorama |
| **Final Panorama Display** | Full-resolution result with size and aspect ratio metadata |
| **Download Button** | Export as PNG or JPEG at full resolution |

---

## Complete File Map

```
CrimeScenceDetecion/
├── app.py                  ← Streamlit UI + user controls
├── run_test.py             ← Headless CLI test runner (no browser)
├── generate_ppt.py         ← Generates a PowerPoint presentation
├── requirements.txt        ← Python dependencies
├── run.sh                  ← One-command startup script
├── real_images/            ← 4 real St. Petersburg river photos
├── sample_images/          ← 4 synthetic cityscape images
└── stitcher/               ← Core stitching library
    ├── pipeline.py         ← Orchestrates all 9 steps
    ├── cylindrical.py      ← Cylindrical warping math
    ├── feature_detector.py ← SIFT / ORB detection
    ├── feature_matcher.py  ← FLANN / BFMatcher + Lowe's ratio test
    ├── homography.py       ← RANSAC Affine / Homography estimation
    ├── ordering.py         ← Graph-based sequence optimizer
    ├── warper.py           ← Warp to canvas + 3 blend modes
    ├── rect_crop.py        ← Largest interior rectangle crop
    └── utils.py            ← Image I/O helpers
```

---

## Why Does the Final Panorama Look So Good?

| Problem | Solution Used |
|:--------|:-------------|
| Wedge / trapezoid distortion | Cylindrical projection + Affine transform (not full homography) |
| Blurry / drifted panorama | Central reference frame — all images aligned to the middle one |
| Wrong photo order | Graph-based inlier-count ordering |
| Visible seams at joins | Multi-band Laplacian pyramid blending |
| Black borders around panorama | O(W×H) largest-rectangle crop algorithm |
| False feature matches | Lowe's ratio test + RANSAC outlier rejection |
