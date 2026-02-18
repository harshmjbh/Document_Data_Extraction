/**
 * Robust page detection: Canny-style edges, morphological closing,
 * contour selection with rectangularity and aspect-ratio filtering.
 * Usage: include in page and worker; call detectPage(imageData) -> quad or null.
 */
(function (global) {
  "use strict";

  function toGrayscale(data, w, h) {
    var out = new Uint8Array(w * h);
    for (var i = 0; i < w * h; i++) {
      var r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      out[i] = (0.299 * r + 0.587 * g + 0.114 * b) | 0;
    }
    return out;
  }

  // 5x5 Gaussian kernel (sigma ~1.4)
  var GAUSS5 = [2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2];
  var GAUSS5_SUM = 159;

  function gaussianBlur(src, w, h) {
    var out = new Uint8Array(w * h);
    for (var y = 2; y < h - 2; y++) {
      for (var x = 2; x < w - 2; x++) {
        var sum = 0;
        for (var dy = -2; dy <= 2; dy++) {
          for (var dx = -2; dx <= 2; dx++) {
            sum += src[(y + dy) * w + (x + dx)] * GAUSS5[(dy + 2) * 5 + (dx + 2)];
          }
        }
        out[y * w + x] = (sum / GAUSS5_SUM) | 0;
      }
    }
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (out[y * w + x] === 0) out[y * w + x] = src[y * w + x];
      }
    }
    return out;
  }

  // Sobel: returns { mag: Uint8Array, dir: Float32Array } (dir in radians)
  var GX = [-1, 0, 1, -2, 0, 2, -1, 0, 1], GY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  function sobel(gray, w, h) {
    var mag = new Uint8Array(w * h);
    var dir = new Float32Array(w * h);
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var sx = 0, sy = 0;
        for (var dy = -1; dy <= 1; dy++) {
          for (var dx = -1; dx <= 1; dx++) {
            var v = gray[(y + dy) * w + (x + dx)];
            var k = (dy + 1) * 3 + (dx + 1);
            sx += v * GX[k];
            sy += v * GY[k];
          }
        }
        var m = Math.sqrt(sx * sx + sy * sy) | 0;
        mag[y * w + x] = m > 255 ? 255 : m;
        dir[y * w + x] = Math.atan2(sy, sx);
      }
    }
    return { mag: mag, dir: dir };
  }

  // Non-maximum suppression: thin edges
  // NMS: keep pixel only if it's max along the gradient direction (perpendicular to edge)
  var DI = [-1 - w, -w, 1 - w, 1, 1 + w, w, -1 + w, -1]; // 8-neighbor offsets
  var NMS_PAIRS = [[7, 3], [0, 4], [1, 5], [2, 6]]; // sector -> (left, right) neighbor indices
  function nonMaxSuppress(mag, dir, w, h) {
    var out = new Uint8Array(w * h);
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var i = y * w + x;
        var m = mag[i];
        if (m === 0) continue;
        var angle = dir[i];
        var sector = ((angle + Math.PI) / (Math.PI / 2)) | 0;
        sector = (sector + 4) % 4;
        if (sector < 0) sector = 0;
        var p = NMS_PAIRS[sector];
        var i1 = i + DI[p[0]], i2 = i + DI[p[1]];
        if (m >= mag[i1] && m >= mag[i2]) out[i] = m;
      }
    }
    return out;
  }

  // Double threshold + hysteresis
  function hysteresis(thin, w, h, low, high) {
    var out = new Uint8Array(w * h);
    var dx = [-1, -1, 0, 1, 1, 1, 0, -1], dy = [0, -1, -1, -1, 0, 1, 1, 1];
    function strong(i) { return thin[i] >= high; }
    function weak(i) { return thin[i] >= low && thin[i] < high; }
    for (var i = 0; i < w * h; i++) if (strong(i)) out[i] = 255;
    var changed = true;
    while (changed) {
      changed = false;
      for (var y = 1; y < h - 1; y++) {
        for (var x = 1; x < w - 1; x++) {
          var idx = y * w + x;
          if (out[idx] !== 0 || !weak(idx)) continue;
          for (var d = 0; d < 8; d++) {
            var nx = x + dx[d], ny = y + dy[d];
            if (out[ny * w + nx] === 255) { out[idx] = 255; changed = true; break; }
          }
        }
      }
    }
    return out;
  }

  // 3x3 binary dilate then erode (closing)
  function morphClose(binary, w, h) {
    var tmp = new Uint8Array(w * h);
    var dx = [-1, -1, 0, 1, 1, 1, 0, -1], dy = [0, -1, -1, -1, 0, 1, 1, 1];
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var i = y * w + x;
        if (binary[i]) { tmp[i] = 255; continue; }
        for (var d = 0; d < 8; d++) {
          if (binary[(y + dy[d]) * w + (x + dx[d])]) { tmp[i] = 255; break; }
        }
      }
    }
    for (var i = 0; i < w * h; i++) binary[i] = 0;
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var i = y * w + x;
        if (!tmp[i]) continue;
        var all = true;
        for (var d = 0; d < 8; d++) {
          if (!tmp[(y + dy[d]) * w + (x + dx[d])]) { all = false; break; }
        }
        if (all) binary[i] = 255;
      }
    }
    return binary;
  }

  function findContours(binary, w, h) {
    var visited = new Uint8Array(w * h);
    var contours = [];
    var dx = [-1, -1, 0, 1, 1, 1, 0, -1], dy = [0, -1, -1, -1, 0, 1, 1, 1];
    function get(x, y) { return (x >= 0 && x < w && y >= 0 && y < h) ? binary[y * w + x] : 0; }
    function floodFill(sx, sy) {
      var pts = [], stack = [[sx, sy]];
      visited[sy * w + sx] = 1;
      while (stack.length) {
        var xy = stack.pop(), x = xy[0], y = xy[1];
        pts.push({ x: x, y: y });
        for (var d = 0; d < 8; d++) {
          var nx = x + dx[d], ny = y + dy[d];
          if (get(nx, ny) && !visited[ny * w + nx]) {
            visited[ny * w + nx] = 1;
            stack.push([nx, ny]);
          }
        }
      }
      return pts;
    }
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (binary[y * w + x] && !visited[y * w + x]) {
          var c = floodFill(x, y);
          if (c.length >= 4) contours.push(c);
        }
      }
    }
    return contours;
  }

  function convexHull(pts) {
    if (pts.length < 3) return pts;
    var pivot = pts[0];
    for (var i = 1; i < pts.length; i++) {
      var p = pts[i];
      if (p.y < pivot.y || (p.y === pivot.y && p.x < pivot.x)) pivot = p;
    }
    var sorted = [];
    for (var i = 0; i < pts.length; i++) {
      if (pts[i] === pivot) continue;
      var p = pts[i];
      sorted.push({ x: p.x, y: p.y, angle: Math.atan2(p.y - pivot.y, p.x - pivot.x) });
    }
    sorted.sort(function (a, b) { return a.angle - b.angle; });
    var hull = [pivot];
    for (var s = 0; s < sorted.length; s++) {
      var p = { x: sorted[s].x, y: sorted[s].y };
      while (hull.length >= 2) {
        var a = hull[hull.length - 2], b = hull[hull.length - 1];
        if ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) <= 0) hull.pop();
        else break;
      }
      hull.push(p);
    }
    return hull;
  }

  function lineDist(p, a, b) {
    var num = Math.abs((b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x);
    return num / (Math.sqrt((b.y - a.y) * (b.y - a.y) + (b.x - a.x) * (b.x - a.x)) || 1);
  }

  function douglasPeucker(pts, epsilon) {
    if (pts.length <= 2) return pts;
    var maxD = 0, index = 0, end = pts.length - 1;
    for (var i = 1; i < end; i++) {
      var d = lineDist(pts[i], pts[0], pts[end]);
      if (d > maxD) { maxD = d; index = i; }
    }
    if (maxD <= epsilon) return [pts[0], pts[end]];
    var left = douglasPeucker(pts.slice(0, index + 1), epsilon);
    var right = douglasPeucker(pts.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }

  function polygonArea(pts) {
    var area = 0, n = pts.length;
    for (var i = 0; i < n; i++)
      area += pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y;
    return Math.abs(area) / 2;
  }

  function boundingRect(pts) {
    var minX = pts[0].x, maxX = pts[0].x, minY = pts[0].y, maxY = pts[0].y;
    for (var i = 1; i < pts.length; i++) {
      if (pts[i].x < minX) minX = pts[i].x;
      if (pts[i].x > maxX) maxX = pts[i].x;
      if (pts[i].y < minY) minY = pts[i].y;
      if (pts[i].y > maxY) maxY = pts[i].y;
    }
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
  }

  function orderCorners(pts) {
    var sum = function (p) { return p.x + p.y; }, diff = function (p) { return p.x - p.y; };
    var tl = pts[0], tr = pts[0], br = pts[0], bl = pts[0];
    for (var i = 1; i < pts.length; i++) {
      if (sum(pts[i]) < sum(tl)) tl = pts[i];
      if (sum(pts[i]) > sum(br)) br = pts[i];
      if (diff(pts[i]) < diff(tr)) tr = pts[i];
      if (diff(pts[i]) > diff(bl)) bl = pts[i];
    }
    return [tl, tr, br, bl];
  }

  function hullToQuad(hull) {
    if (hull.length < 4) return null;
    var tl = hull[0], tr = hull[0], br = hull[0], bl = hull[0];
    for (var i = 1; i < hull.length; i++) {
      var p = hull[i];
      if (p.x + p.y < tl.x + tl.y) tl = p;
      if (p.x + p.y > br.x + br.y) br = p;
      if (p.x - p.y < tr.x - tr.y) tr = p;
      if (p.x - p.y > bl.x - bl.y) bl = p;
    }
    return orderCorners([tl, tr, br, bl]);
  }

  /**
   * Robust document quad detection.
   * @param {ImageData} imageData
   * @returns {Array<{x,y}>|null} 4 corners TL, TR, BR, BL or null
   */
  function detectPage(imageData) {
    var w = imageData.width, h = imageData.height;
    var gray = toGrayscale(imageData.data, w, h);
    var blurred = gaussianBlur(gray, w, h);
    var sobelOut = sobel(blurred, w, h);
    var thin = nonMaxSuppress(sobelOut.mag, sobelOut.dir, w, h);
    var low = 25, high = 60;
    var edges = hysteresis(thin, w, h, low, high);
    morphClose(edges, w, h);
    var contours = findContours(edges, w, h);

    var minArea = w * h * 0.04;
    var bestQuad = null;
    var bestScore = 0;

    for (var c = 0; c < contours.length; c++) {
      var hull = convexHull(contours[c]);
      var area = polygonArea(hull);
      if (area < minArea) continue;

      var rect = boundingRect(hull);
      var rectArea = rect.w * rect.h;
      if (rectArea < 1) continue;
      var aspect = rect.w > rect.h ? rect.w / rect.h : rect.h / rect.w;
      if (aspect > 6 || aspect < 0.15) continue;

      var peri = 0;
      for (var i = 0; i < hull.length; i++)
        peri += Math.hypot(hull[(i + 1) % hull.length].x - hull[i].x, hull[(i + 1) % hull.length].y - hull[i].y);
      var approx = douglasPeucker(hull, 0.025 * peri);

      var quad = null;
      if (approx.length === 4) {
        quad = orderCorners(approx);
      } else {
        quad = hullToQuad(hull);
      }
      if (!quad || polygonArea(quad) < minArea) continue;

      var quadArea = polygonArea(quad);
      var rectangularity = quadArea / rectArea;
      var score = quadArea * (0.5 + 0.5 * rectangularity);

      if (score > bestScore) {
        bestScore = score;
        bestQuad = quad;
      }
    }
    return bestQuad;
  }

  var root = typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : global;
  root.detectPage = detectPage;
})(typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : this);
