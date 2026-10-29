// handTracker.js
// Classical (non-deep-learning) hand detection with OpenCV.js.
//
// Pipeline, one frame at a time:
//   1. RGBA frame  -> YCrCb colour space
//   2. Skin mask   = pixels whose Cr and Cb fall inside a range   (cv.inRange)
//   3. Clean mask  = blur + morphological opening and closing     (cv.morphologyEx)
//   4. Hand        = largest contour in the mask                  (cv.findContours)
//   5. Position    = centroid from image moments                  (cv.moments)
//   6. Fingers     = convex hull + convexity defects              (cv.convexHull, cv.convexityDefects)
//   7. Tilt        = orientation from central moments (mu20, mu02, mu11)
//
// Every stage is kept as a Mat so the UI can show each step (useful in the viva).

export const DEFAULT_PARAMS = {
  crMin: 135, crMax: 180,   // classic skin range (Chai & Ngan, 1999), slightly widened
  cbMin: 85,  cbMax: 135,
  kernelSize: 5,            // morphology kernel (pixels)
  minAreaRatio: 0.02,       // ignore blobs smaller than 2% of the region of interest
  defectDepthRatio: 0.18,   // a valley must be at least 18% of hand height deep
  maxDefectAngle: 90,       // ...and narrower than 90 degrees to count as a gap between fingers
};

export function createTracker(cv) {
  const m = {
    ycrcb: new cv.Mat(), rgb: new cv.Mat(),
    rawMask: new cv.Mat(), mask: new cv.Mat(),
    low: null, high: null, kernel: null, kernelSize: 0,
  };

  function getKernel(size) {
    if (m.kernelSize !== size) {
      if (m.kernel) m.kernel.delete();
      m.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
      m.kernelSize = size;
    }
    return m.kernel;
  }

  function setRange(p) {
    if (m.low) { m.low.delete(); m.high.delete(); }
    m.low = new cv.Mat(m.ycrcb.rows, m.ycrcb.cols, m.ycrcb.type(), [0, p.crMin, p.cbMin, 0]);
    m.high = new cv.Mat(m.ycrcb.rows, m.ycrcb.cols, m.ycrcb.type(), [255, p.crMax, p.cbMax, 255]);
  }

  // Angle (degrees) at point b in the triangle a-b-c, via the cosine rule.
  function angleAt(a, b, c) {
    const ab = Math.hypot(a.x - b.x, a.y - b.y);
    const cb = Math.hypot(c.x - b.x, c.y - b.y);
    const ac = Math.hypot(a.x - c.x, a.y - c.y);
    const cos = (ab * ab + cb * cb - ac * ac) / (2 * ab * cb + 1e-6);
    return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
  }

  const pt = (contour, i) => ({ x: contour.data32S[i * 2], y: contour.data32S[i * 2 + 1] });

  /**
   * src: CV_8UC4 (RGBA) Mat of the region of interest.
   * Returns { found, fingers, cx, cy, angle, area, contourPts, hullPts, tips, valleys } in pixel units.
   */
  function process(src, p = DEFAULT_PARAMS) {
    // 1. Colour conversion. YCrCb separates brightness (Y) from colour (Cr, Cb),
    //    so the skin test ignores Y and copes better with lighting changes.
    cv.cvtColor(src, m.rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(m.rgb, m.ycrcb, cv.COLOR_RGB2YCrCb);

    // 2. Threshold on Cr and Cb only (Y range is 0..255).
    setRange(p);
    cv.inRange(m.ycrcb, m.low, m.high, m.rawMask);

    // 3. Noise removal: median blur, then opening (removes specks) and closing (fills holes).
    const k = getKernel(p.kernelSize);
    cv.medianBlur(m.rawMask, m.mask, 5);
    cv.morphologyEx(m.mask, m.mask, cv.MORPH_OPEN, k);
    cv.morphologyEx(m.mask, m.mask, cv.MORPH_CLOSE, k, new cv.Point(-1, -1), 2);

    return analyseMask(m.mask, p);
  }

  /** Steps 4-7, on any binary mask (also used for grayscale datasets with Otsu). */
  function analyseMask(mask, p = DEFAULT_PARAMS) {
    const result = { found: false, fingers: 0, cx: 0, cy: 0, angle: 0, area: 0,
                     contourPts: [], hullPts: [], tips: [], valleys: [] };
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    const work = mask.clone();
    cv.findContours(work, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // 4. Pick the largest blob: we assume the hand is the biggest skin region in the box.
    let best = -1, bestArea = 0;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const a = cv.contourArea(c);
      if (a > bestArea) { bestArea = a; best = i; }
      c.delete();
    }
    const minArea = p.minAreaRatio * mask.rows * mask.cols;
    if (best < 0 || bestArea < minArea) {
      contours.delete(); hierarchy.delete(); work.delete();
      return result;
    }

    const cnt = contours.get(best);
    result.found = true;
    result.area = bestArea;
    for (let i = 0; i < cnt.rows; i++) result.contourPts.push(pt(cnt, i));

    // 5. Centroid from spatial moments: cx = m10/m00, cy = m01/m00.
    const mo = cv.moments(cnt, false);
    result.cx = mo.m10 / mo.m00;
    result.cy = mo.m01 / mo.m00;

    // 7. Orientation from central moments: theta = 0.5 * atan2(2*mu11, mu20 - mu02).
    //    Converted so that 0 = hand pointing straight up, positive = tilted right (in the image).
    const theta = 0.5 * Math.atan2(2 * mo.mu11, mo.mu20 - mo.mu02);
    let deg = (theta * 180) / Math.PI;          // angle of main axis from +x axis
    deg = deg > 0 ? deg - 90 : deg + 90;         // relative to vertical
    result.angle = deg;

    // 6. Convex hull (as point indices, needed for convexityDefects).
    const hullIdx = new cv.Mat();
    cv.convexHull(cnt, hullIdx, false, false);
    for (let i = 0; i < hullIdx.rows; i++) result.hullPts.push(pt(cnt, hullIdx.data32S[i]));

    const rect = cv.boundingRect(cnt);
    const minDepth = p.defectDepthRatio * rect.height;
    let gaps = 0;

    if (hullIdx.rows > 3) {
      const defects = new cv.Mat();
      try {
        cv.convexityDefects(cnt, hullIdx, defects);
        // Each defect row: [startIdx, endIdx, farthestIdx, depth*256]
        for (let i = 0; i < defects.rows; i++) {
          const d = defects.data32S.subarray(i * 4, i * 4 + 4);
          const s = pt(cnt, d[0]), e = pt(cnt, d[1]), f = pt(cnt, d[2]);
          const depth = d[3] / 256;
          // A gap between two fingers is deep and narrow, and its valley sits
          // above the bottom of the hand (rules out the wrist corners).
          if (depth > minDepth && angleAt(s, f, e) < p.maxDefectAngle && f.y < rect.y + rect.height * 0.85) {
            gaps++;
            result.valleys.push(f);
            result.tips.push(s, e);
          }
        }
      } catch (err) {
        // convexityDefects can throw on self-intersecting contours; treat as no gaps.
      }
      defects.delete();
    }

    if (gaps > 0) {
      result.fingers = Math.min(5, gaps + 1);   // n gaps between fingers -> n+1 fingers
    } else {
      // No gaps: either a fist (0) or one raised finger (1).
      // A single finger makes the shape tall compared with its width.
      const hullArea = cv.contourArea(matFromPts(result.hullPts));
      const solidity = bestArea / (hullArea + 1e-6);
      const tall = rect.height / rect.width > 1.6;
      result.fingers = tall && solidity < 0.85 ? 1 : 0;
    }
    // De-duplicate fingertip markers (neighbouring defects share endpoints).
    result.tips = dedupe(result.tips, rect.height * 0.08);

    hullIdx.delete(); cnt.delete(); contours.delete(); hierarchy.delete(); work.delete();
    return result;
  }

  let tmpHull = null;
  function matFromPts(pts) {
    if (tmpHull) tmpHull.delete();
    tmpHull = cv.matFromArray(pts.length, 1, cv.CV_32SC2, pts.flatMap((q) => [q.x, q.y]));
    return tmpHull;
  }

  function dedupe(points, minDist) {
    const out = [];
    for (const q of points) if (!out.some((o) => Math.hypot(o.x - q.x, o.y - q.y) < minDist)) out.push(q);
    return out;
  }

  /** Learn a skin range from a patch: mean +/- k standard deviations of Cr and Cb. */
  function calibrate(src, k = 2.5) {
    cv.cvtColor(src, m.rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(m.rgb, m.ycrcb, cv.COLOR_RGB2YCrCb);
    const mean = new cv.Mat(), std = new cv.Mat();
    cv.meanStdDev(m.ycrcb, mean, std);
    const mu = mean.data64F, sd = std.data64F;
    mean.delete(); std.delete();
    const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
    return {
      crMin: clamp(mu[1] - k * sd[1] - 4), crMax: clamp(mu[1] + k * sd[1] + 4),
      cbMin: clamp(mu[2] - k * sd[2] - 4), cbMax: clamp(mu[2] + k * sd[2] + 4),
    };
  }

  return { process, analyseMask, calibrate, mats: m };
}
