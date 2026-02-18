/**
 * Page detection Web Worker. Uses robust Canny + morphology pipeline from page-detection.js.
 * Receives { width, height, data } (RGBA), returns { quad } (4 corners TL,TR,BR,BL or null).
 */
importScripts("page-detection.js");

self.onmessage = function (e) {
  var width = e.data.width, height = e.data.height, data = e.data.data;
  if (!data || !width || !height) {
    self.postMessage({ quad: null });
    return;
  }
  var imageData = new ImageData(new Uint8ClampedArray(data), width, height);
  var quad = detectPage(imageData);
  self.postMessage({ quad: quad });
};
