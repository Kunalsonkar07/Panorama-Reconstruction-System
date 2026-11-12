"""
Core panorama stitching pipeline.

Features:
─────────
1. Cylindrical Projection:
   Maps rotation around vertical camera axis to pure horizontal translation.
   Prevents edge curvature and distortion on wide panoramas.

2. Affine / Rigid Transformation:
   Prevents projective division divergence (wedge / horn distortion).
   Guarantees uniform height and width from left to right.

3. Central Reference Frame:
   Aligns all images relative to the middle image of the sequence.
   Symmetrically balances alignment errors with zero cumulative blur.

4. Gain / Exposure Compensation:
   Normalizes channel-wise brightness across overlaps to eliminate exposure seams.

5. Seamless Blending:
   Multi-band Laplacian pyramid or distance-transform feathering.

6. Automatic Rectangular Cropping:
   Extracts the largest fully-valid interior rectangle.
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict

from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .homography import HomographyEstimator
from .warper import ImageWarper
from .ordering import ImageOrderer
from .cylindrical import cylindrical_warp, estimate_focal_length, strip_black_cols


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    img_idx_a: int
    img_idx_b: int
    num_matches: int
    num_inliers: int
    homography: Optional[np.ndarray]
    match_vis: Optional[np.ndarray] = None


@dataclass
class StitchResult:
    panorama: Optional[np.ndarray]
    stitch_order: List[int]
    pair_results: List[PairResult]
    failed_images: List[int]
    log: List[str] = field(default_factory=list)
    intermediate_panoramas: List[np.ndarray] = field(default_factory=list)
    naive_panorama: Optional[np.ndarray] = None
    timings: Dict[str, float] = field(default_factory=dict)
    total_keypoints: int = 0
    avg_inliers: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class StitchingPipeline:
    """
    High-accuracy image stitching pipeline.
    """

    def __init__(
        self,
        method: str = "SIFT",
        max_features: int = 5000,
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 4.0,
        min_matches: int = 10,
        min_inliers: int = 8,
        transform_type: str = "affine",  # "affine" (uniform width/height) or "homography"
        blend_mode: str = "multiband",    # "multiband", "feather", "simple"
        use_cylindrical: bool = True,
        focal_length: float = 0.0,
        crop_mode: str = "rect",         # "rect", "tight", "none"
        engine: str = "custom",          # "custom" or "opencv"
        compensate_exposure: bool = True,
        compute_comparison: bool = True,
    ):
        self.method = method.upper()
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.transform_type = transform_type.lower()
        self.blend_mode = blend_mode.lower()
        self.use_cylindrical = use_cylindrical
        self.focal_length = focal_length
        self.crop_mode = crop_mode
        self.engine = engine.lower()
        self.compensate_exposure = compensate_exposure
        self.compute_comparison = compute_comparison

        self.detector = FeatureDetector(method=method, max_features=max_features)
        self.matcher = FeatureMatcher(
            method=method,
            ratio_thresh=ratio_thresh,
            min_good_matches=min_matches,
        )
        self.estimator = HomographyEstimator(
            ransac_reproj_thresh=ransac_thresh,
            min_inliers=min_inliers,
            transform_type=self.transform_type,
        )
        self.warper = ImageWarper(
            blend_mode=self.blend_mode,
            compensate_exposure=self.compensate_exposure,
        )

    # ─────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────

    def stitch(
        self,
        images: List[np.ndarray],
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> StitchResult:
        log: List[str] = []
        n = len(images)
        timings: Dict[str, float] = {}

        def _cb(msg: str, frac: float):
            log.append(msg)
            if progress_cb:
                progress_cb(msg, frac)

        if n == 0:
            return StitchResult(None, [], [], [], log)
        if n == 1:
            return StitchResult(images[0].copy(), [0], [], [], log)

        # ── Optional: OpenCV Native Engine ───────────────────────────
        if self.engine == "opencv":
            t0 = time.time()
            _cb("Running OpenCV Native Stitcher …", 0.3)
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, pano = stitcher.stitch(images)
            timings["opencv_native"] = time.time() - t0
            if status == cv2.Stitcher_OK:
                _cb(f"OpenCV stitch succeeded ({pano.shape[1]}×{pano.shape[0]} px)", 1.0)
                if self.crop_mode == "rect":
                    pano = self.warper.crop_largest_rectangle(pano)
                elif self.crop_mode == "tight":
                    pano = self.warper.crop_to_valid(pano)
                return StitchResult(
                    panorama=pano,
                    stitch_order=list(range(n)),
                    pair_results=[],
                    failed_images=[],
                    log=log,
                    intermediate_panoramas=[pano],
                    timings=timings,
                )
            else:
                _cb("OpenCV stitch failed. Falling back to custom pipeline …", 0.4)

        # ── Step 1: Cylindrical Projection ───────────────────────────
        t0_proj = time.time()
        if self.use_cylindrical:
            _cb("Applying cylindrical projection to equalize scale and translation …", 0.05)
            f = self.focal_length if self.focal_length > 0 else estimate_focal_length(images[0])
            _cb(f"  Focal length: {f:.1f} px", 0.07)
            work_images = []
            for i, img in enumerate(images):
                w_img = cylindrical_warp(img, f)
                w_img = strip_black_cols(w_img)
                work_images.append(w_img)
                _cb(f"  Image {i+1}/{n} projected ({img.shape[1]}×{img.shape[0]} → {w_img.shape[1]}×{w_img.shape[0]})",
                    0.07 + 0.08 * (i + 1) / n)
        else:
            work_images = [img.copy() for img in images]
        timings["cylindrical_projection"] = time.time() - t0_proj

        # ── Step 2: Feature Detection ────────────────────────────────
        t0_feat = time.time()
        _cb(f"Detecting features with {self.method} …", 0.16)
        keypoints: List = []
        descriptors: List = []
        total_kps = 0
        for i, img in enumerate(work_images):
            kp, desc = self.detector.detect_and_compute(img)
            keypoints.append(kp)
            descriptors.append(desc)
            total_kps += len(kp)
            _cb(f"  Image {i+1}/{n}: {len(kp)} keypoints", 0.16 + 0.10 * (i + 1) / n)
        timings["feature_detection"] = time.time() - t0_feat

        # ── Step 3: Pairwise Matching & Graph Ordering ───────────────
        t0_match = time.time()
        _cb("Matching features across all image pairs …", 0.28)
        orderer = ImageOrderer(n)
        pair_results: List[PairResult] = []

        total_pairs = n * (n - 1) // 2
        pair_idx = 0
        inlier_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                matches = self.matcher.match(descriptors[i], descriptors[j])
                H, mask = None, None
                inliers = 0

                if self.matcher.is_good_match(matches):
                    H, mask = self.estimator.estimate(
                        keypoints[i], keypoints[j], matches
                    )
                    inliers = self.estimator.count_inliers(mask)

                orderer.record_match(i, j, inliers if H is not None else 0)
                inlier_sum += inliers

                match_vis = None
                if matches:
                    try:
                        match_vis = self.matcher.draw_matches(
                            work_images[i], keypoints[i],
                            work_images[j], keypoints[j],
                            matches,
                        )
                    except Exception:
                        pass

                pair_results.append(PairResult(i, j, len(matches), inliers, H, match_vis))
                pair_idx += 1
                frac = 0.28 + 0.20 * pair_idx / total_pairs
                _cb(
                    f"  Pair ({i+1},{j+1}): {len(matches)} matches, {inliers} inliers {'✓' if H is not None else '✗'}",
                    frac,
                )
        timings["feature_matching"] = time.time() - t0_match

        # ── Step 4: Determine Optimal Stitching Order ────────────────
        t0_align = time.time()
        stitch_order = orderer.compute_order()
        _cb(f"Optimal stitching sequence: {[o+1 for o in stitch_order]}", 0.50)

        # ── Step 5: Compute Transformations to Center Reference ──────
        _cb("Aligning images relative to central reference frame …", 0.55)
        ordered_work = [work_images[idx] for idx in stitch_order]
        ordered_kp = [keypoints[idx] for idx in stitch_order]
        ordered_desc = [descriptors[idx] for idx in stitch_order]

        n_ord = len(stitch_order)
        pair_transforms: List[Optional[np.ndarray]] = []
        failed_transitions: List[int] = []

        for i in range(n_ord - 1):
            matches = self.matcher.match(ordered_desc[i], ordered_desc[i + 1])
            H, mask = self.estimator.estimate(ordered_kp[i], ordered_kp[i + 1], matches)
            if H is not None:
                pair_transforms.append(H)
                inliers = self.estimator.count_inliers(mask)
                _cb(f"  Step {i+1}→{i+2}: verified with {inliers} inliers.", 0.55 + 0.15 * (i + 1) / n_ord)
            else:
                pair_transforms.append(None)
                failed_transitions.append(i + 1)
                _cb(f"  ✗ Weak match between step {i+1} and {i+2}.", 0.55 + 0.15 * (i + 1) / n_ord)

        # Reference image: center of the chain
        mid = n_ord // 2
        _cb(f"Using Image {stitch_order[mid]+1} as central reference frame (minimizes distortion).", 0.72)

        H_to_ref: List[Optional[np.ndarray]] = [None] * n_ord
        H_to_ref[mid] = np.eye(3, dtype=np.float64)

        # Outward right (mid -> end)
        for i in range(mid + 1, n_ord):
            T_prev = pair_transforms[i - 1]
            if T_prev is not None and H_to_ref[i - 1] is not None:
                H_to_ref[i] = H_to_ref[i - 1] @ T_prev
            else:
                failed_transitions.append(i)

        # Outward left (mid -> 0)
        for i in range(mid - 1, -1, -1):
            T_curr = pair_transforms[i]
            if T_curr is not None and H_to_ref[i + 1] is not None:
                try:
                    H_to_ref[i] = H_to_ref[i + 1] @ np.linalg.inv(T_curr)
                except np.linalg.LinAlgError:
                    failed_transitions.append(i)
            else:
                failed_transitions.append(i)

        # Filter valid images
        valid_indices = [i for i, H in enumerate(H_to_ref) if H is not None]
        failed_imgs = [stitch_order[i] for i, H in enumerate(H_to_ref) if H is None]

        if not valid_indices:
            _cb("Stitching failed — could not align image sequence.", 1.0)
            return StitchResult(None, stitch_order, pair_results, stitch_order, log)

        valid_images = [ordered_work[i] for i in valid_indices]
        valid_transforms = [H_to_ref[i] for i in valid_indices]
        timings["global_alignment"] = time.time() - t0_align

        # ── Step 6: Warping, Gain Compensation & Seamless Blending ───
        t0_warp = time.time()
        _cb(f"Warping and blending {len(valid_images)} images ({self.blend_mode} blend, exposure comp={self.compensate_exposure}) …", 0.82)
        ref_in_valid = valid_indices.index(mid) if mid in valid_indices else 0
        panorama, intermediates = self.warper.blend_multiple(
            valid_images,
            valid_transforms,
            crop_mode=self.crop_mode,
            ref_idx=ref_in_valid,
        )
        timings["warping_and_blending"] = time.time() - t0_warp

        # ── Step 7: Optional Naive Planar Baseline (for comparison) ──
        naive_pano = None
        if self.compute_comparison and n >= 2:
            _cb("Generating baseline naive planar homography for distortion comparison …", 0.94)
            naive_pano = self._build_naive_baseline(images, stitch_order)

        h, w = panorama.shape[:2]
        _cb(f"Complete! Panorama generated: {w} × {h} px (uniform height).", 1.0)

        avg_inliers = (inlier_sum / max(total_pairs, 1)) if total_pairs > 0 else 0.0

        return StitchResult(
            panorama=panorama,
            stitch_order=stitch_order,
            pair_results=pair_results,
            failed_images=failed_imgs,
            log=log,
            intermediate_panoramas=intermediates,
            naive_panorama=naive_pano,
            timings=timings,
            total_keypoints=total_kps,
            avg_inliers=avg_inliers,
        )

    # ─────────────────────────────────────────────────────────────────
    # Naive Comparison Baseline
    # ─────────────────────────────────────────────────────────────────

    def _build_naive_baseline(
        self,
        raw_images: List[np.ndarray],
        ordered_indices: List[int],
    ) -> Optional[np.ndarray]:
        """
        Builds a naive sequential projective homography panorama (planar, no cylindrical
        projection, chained left-to-right, uncropped).
        This illustrates the classic wedge / trumpet distortion for visual comparison.
        """
        try:
            ordered_raw = [raw_images[idx] for idx in ordered_indices]
            n = len(ordered_raw)
            if n < 2:
                return None

            # Detect features on flat raw images
            kps, descs = [], []
            for img in ordered_raw:
                kp, desc = self.detector.detect_and_compute(img)
                kps.append(kp)
                descs.append(desc)

            # Homography estimator (pure projective 8-DOF, not affine)
            homog_estimator = HomographyEstimator(
                ransac_reproj_thresh=self.ransac_thresh,
                min_inliers=self.min_inliers,
                transform_type="homography",
            )

            # Chain sequentially to image 0
            H_to_first = [np.eye(3, dtype=np.float64)]
            for i in range(n - 1):
                matches = self.matcher.match(descs[i], descs[i + 1])
                H, mask = homog_estimator.estimate(kps[i], kps[i + 1], matches)
                if H is None:
                    return None
                H_to_first.append(H_to_first[-1] @ H)

            # Warp to canvas with simple overlay to show raw geometric distortion
            corners_all = []
            for i, img in enumerate(ordered_raw):
                h, w = img.shape[:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                warped_c = cv2.perspectiveTransform(corners, H_to_first[i])
                corners_all.append(warped_c)

            corners_all = np.concatenate(corners_all, axis=0)
            x_min, y_min = np.int32(corners_all.min(axis=0).ravel()) - 2
            x_max, y_max = np.int32(corners_all.max(axis=0).ravel()) + 2

            cw = min(int(x_max - x_min), 12000)
            ch = min(int(y_max - y_min), 6000)
            if cw <= 0 or ch <= 0:
                return None

            T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
            naive_canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
            for i, img in enumerate(ordered_raw):
                warped = cv2.warpPerspective(img, T @ H_to_first[i], (cw, ch))
                mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 1
                naive_canvas[mask] = warped[mask]

            return naive_canvas
        except Exception:
            return None
