// wasm_test.js
// Minimal Node.js test harness for the scanner WebAssembly module.
//
// This does NOT load any real image files. Instead, it:
//   - creates a small synthetic RGBA image buffer,
//   - calls _detect_document via the C API,
//   - calls _warp_document to resize it,
//   - logs the returned corners and a checksum of the warped buffer.

const createScannerModule = require("./scanner.js");

async function main() {
  const Module = await createScannerModule();

  const width = 4;
  const height = 4;
  const numPixels = width * height;
  const numBytes = numPixels * 4;

  // Create a simple test image: horizontal gradient in red channel.
  const hostBuffer = new Uint8Array(numBytes);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      hostBuffer[idx + 0] = (x / (width - 1)) * 255; // R
      hostBuffer[idx + 1] = 0;                       // G
      hostBuffer[idx + 2] = 0;                       // B
      hostBuffer[idx + 3] = 255;                     // A
    }
  }

  // Allocate input buffer in WASM heap.
  const inPtr = Module._malloc(numBytes);
  Module.HEAPU8.set(hostBuffer, inPtr);

  // Allocate outputs for detect_document: 8 floats for corners, 1 float for confidence.
  const cornersBytes = 8 * 4;
  const confBytes = 4;
  const cornersPtr = Module._malloc(cornersBytes);
  const confPtr = Module._malloc(confBytes);

  const found = Module._detect_document(inPtr, width, height, cornersPtr, confPtr);

  console.log("detect_document returned:", found);
  if (!found) {
    console.log("No document detected.");
    Module._free(inPtr);
    Module._free(cornersPtr);
    Module._free(confPtr);
    return;
  }

  const cornersView = new Float32Array(Module.HEAPF32.buffer, cornersPtr, 8);
  const confView = new Float32Array(Module.HEAPF32.buffer, confPtr, 1);

  console.log("confidence:", confView[0]);
  console.log("corners (TL,TR,BR,BL):");
  for (let i = 0; i < 4; i++) {
    console.log(
      `  (${cornersView[i * 2 + 0].toFixed(2)}, ${cornersView[i * 2 + 1].toFixed(
        2
      )})`
    );
  }

  // Now call warp_document to resize to 2x2.
  const outWidth = 2;
  const outHeight = 2;
  const outBytes = outWidth * outHeight * 4;
  const outPtr = Module._malloc(outBytes);

  const ok = Module._warp_document(
    inPtr,
    width,
    height,
    cornersPtr,
    outPtr,
    outWidth,
    outHeight
  );

  console.log("warp_document returned:", ok);
  if (!ok) {
    console.log("warp_document failed.");
    Module._free(inPtr);
    Module._free(cornersPtr);
    Module._free(confPtr);
    Module._free(outPtr);
    return;
  }

  const outView = new Uint8Array(Module.HEAPU8.buffer, outPtr, outBytes);
  // Compute a simple checksum so we can see that something happened.
  let checksum = 0;
  for (let i = 0; i < outView.length; i++) {
    checksum = (checksum + outView[i]) >>> 0;
  }

  console.log("Warped buffer length:", outView.length);
  console.log("Warped buffer checksum:", checksum);

  // Clean up.
  Module._free(inPtr);
  Module._free(cornersPtr);
  Module._free(confPtr);
  Module._free(outPtr);
}

main().catch((err) => {
  console.error("wasm_test error:", err);
  process.exit(1);
});

