
import streamlit as st
import cv2
import hashlib
import numpy as np
import os
import json
import time
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Image Security — Tamper Detector",
    page_icon="🔐",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────
SECURED_DIR       = "secured_images"          # folder produced by your pipeline
HASH_REGISTRY_JSON = "hash_registry.json"    # optional: pre-saved registry file


# ─────────────────────────────────────────────────────────────────────────────
# DCT WATERMARK CLASS  (exact copy from your Colab pipeline)
# ─────────────────────────────────────────────────────────────────────────────
class DCTWatermark:
    BLOCK    = 8
    STRENGTH = 15.0
    ROW, COL = 4, 4

    def _text_to_bits(self, text: str) -> list:
        bits = []
        for ch in text:
            byte = ord(ch)
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def _bits_to_text(self, bits: list) -> str:
        chars = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for b in bits[i:i+8]:
                byte = (byte << 1) | b
            if byte == 0:
                break
            chars.append(chr(byte))
        return "".join(chars)

    def embed(self, image_bgr: np.ndarray, image_id: str) -> np.ndarray:
        gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w    = gray.shape
        wm      = gray.copy()
        bits    = self._text_to_bits(image_id)
        bit_idx = 0

        for r in range(0, h - self.BLOCK + 1, self.BLOCK):
            for c in range(0, w - self.BLOCK + 1, self.BLOCK):
                if bit_idx >= len(bits):
                    break
                block     = wm[r:r+self.BLOCK, c:c+self.BLOCK]
                dct_block = cv2.dct(block)
                if bits[bit_idx] == 1:
                    dct_block[self.ROW, self.COL] += self.STRENGTH
                else:
                    dct_block[self.ROW, self.COL] -= self.STRENGTH
                wm[r:r+self.BLOCK, c:c+self.BLOCK] = cv2.idct(dct_block)
                bit_idx += 1

        wm     = np.clip(wm, 0, 255).astype(np.uint8)
        result = image_bgr.copy()
        result[:, :, 1] = wm
        return result

    def extract(self, image_bgr: np.ndarray, num_chars: int) -> str:
        gray   = image_bgr[:, :, 1].astype(np.float64)
        h, w   = gray.shape
        bits   = []
        needed = num_chars * 8

        for r in range(0, h - self.BLOCK + 1, self.BLOCK):
            for c in range(0, w - self.BLOCK + 1, self.BLOCK):
                if len(bits) >= needed:
                    break
                block     = gray[r:r+self.BLOCK, c:c+self.BLOCK]
                dct_block = cv2.dct(block)
                bits.append(1 if dct_block[self.ROW, self.COL] > 0 else 0)

        return self._bits_to_text(bits)

    @staticmethod
    def compute_hash(image_bgr: np.ndarray) -> str:
        return hashlib.sha256(image_bgr.tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# HASH REGISTRY BUILDER  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_hash_registry(secured_dir: str, json_path: str) -> dict:
    """
    Priority order:
      1. Load from hash_registry.json if it exists (fast, recommended).
      2. Scan secured_images/ folder and recompute hashes (slower first run).
    """
    # ── Option 1: load pre-saved JSON ────────────────────────────────────────
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            registry = json.load(f)
        return registry

    # ── Option 2: scan folder and hash every image ────────────────────────────
    registry = {}
    secured  = Path(secured_dir)
    if not secured.exists():
        return registry

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for img_path in secured.rglob("*"):
        if img_path.suffix.lower() not in valid_ext:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Derive image_id from file path:
        # secured_images/PCOS/wm_Image_001.png  → "PCOS_Image_001"
        # secured_images/Non-PCOS/wm_abc.png    → "Non-PCOS_abc"
        parent_name = img_path.parent.name          # "PCOS" or "Non-PCOS"
        stem        = img_path.stem.replace("wm_", "", 1)
        image_id    = f"{parent_name}_{stem}"

        registry[image_id] = DCTWatermark.compute_hash(img)

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# CORE DETECTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def detect_tampering(img_bgr: np.ndarray,
                      registry: dict) -> dict:
    """
    Returns a dict with:
        status       : "INTACT" | "TAMPERED" | "UNKNOWN"
        current_hash : str
        stored_hash  : str | None
        image_id     : str | None
    """
    current_hash = DCTWatermark.compute_hash(img_bgr)

    # Search registry for a matching or known hash
    for img_id, stored_hash in registry.items():
        if stored_hash == current_hash:
            return {
                "status":       "INTACT",
                "current_hash": current_hash,
                "stored_hash":  stored_hash,
                "image_id":     img_id,
            }

    # Hash not found — check if image_id can be guessed from pixel content
    # by trying every entry (hash mismatch = tampered if ID is recognisable)
    # For a definitive TAMPERED result we need to know which entry it corresponds
    # to. We do a best-effort: no match in registry → UNKNOWN by default.
    return {
        "status":       "UNKNOWN",
        "current_hash": current_hash,
        "stored_hash":  None,
        "image_id":     None,
    }


def detect_with_known_id(img_bgr: np.ndarray,
                          image_id: str,
                          registry: dict) -> dict:
    """Same logic but the user explicitly supplies the image ID."""
    current_hash = DCTWatermark.compute_hash(img_bgr)
    stored_hash  = registry.get(image_id)

    if stored_hash is None:
        status = "UNKNOWN"
    elif current_hash == stored_hash:
        status = "INTACT"
    else:
        status = "TAMPERED"

    return {
        "status":       status,
        "current_hash": current_hash,
        "stored_hash":  stored_hash,
        "image_id":     image_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main container */
.main { max-width: 780px; }

/* Result banners */
.result-intact {
    background: #0d3d20;
    border: 1.5px solid #22c55e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-tampered {
    background: #3d0d0d;
    border: 1.5px solid #ef4444;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-unknown {
    background: #2d2a0d;
    border: 1.5px solid #eab308;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.result-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 6px;
}
.result-sub {
    font-size: 0.95rem;
    opacity: 0.75;
    margin: 0;
}
.hash-box {
    background: #111;
    border: 0.5px solid #333;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: monospace;
    font-size: 12px;
    word-break: break-all;
    margin-top: 8px;
    color: #aaa;
}
.section-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #666;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — REGISTRY STATUS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 PCOS Security Tool")
    st.markdown("---")

    # Load registry
    with st.spinner("Loading hash registry…"):
        registry = build_hash_registry(SECURED_DIR, HASH_REGISTRY_JSON)

    if registry:
        st.success(f"Registry loaded\n\n**{len(registry):,}** image hashes")
    else:
        st.warning(
            "No registry found.\n\n"
            "Either:\n"
            "- Place `hash_registry.json` here, **or**\n"
            "- Place your `secured_images/` folder here"
        )

    st.markdown("---")
    st.markdown("**Detection mode**")
    mode = st.radio(
        "",
        ["Auto-detect (hash scan)", "Manual ID entry"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>Based on Eswaraiah & Sreenivasa Reddy — "
        "Medical Image Watermarking for Tamper Detection in ROI</small>",
        unsafe_allow_html=True,
    )

    # ── Export registry button ────────────────────────────────────────────────
    if registry:
        st.markdown("---")
        registry_json = json.dumps(registry, indent=2)
        st.download_button(
            "💾 Export hash_registry.json",
            data=registry_json,
            file_name="hash_registry.json",
            mime="application/json",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Medical Image Tamper Detector")
st.markdown(
    "Upload a watermarked PCOS ultrasound image. "
    "The app checks its SHA-256 hash against the registry built during watermarking."
)

# ── Image ID input (manual mode) ─────────────────────────────────────────────
manual_id = None
if mode == "Manual ID entry":
    manual_id = st.text_input(
        "Image ID",
        placeholder="e.g. PCOS_Image_003 or Non-PCOS_Image_021",
        help="Enter the image ID that was used when watermarking (format: ClassName_stem)",
    )

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    help="Any image — if it was watermarked by your pipeline the hash will match.",
)

if uploaded is not None:
    # Read image bytes → numpy BGR (same as cv2.imread)
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode image. Please try a different file.")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Run detection ─────────────────────────────────────────────────────────
    with st.spinner("Analysing image…"):
        time.sleep(0.4)   # brief pause so spinner is visible
        if not registry:
            result = {
                "status":       "UNKNOWN",
                "current_hash": DCTWatermark.compute_hash(img_bgr),
                "stored_hash":  None,
                "image_id":     None,
            }
        elif mode == "Manual ID entry" and manual_id:
            result = detect_with_known_id(img_bgr, manual_id.strip(), registry)
        else:
            result = detect_tampering(img_bgr, registry)

    status = result["status"]

    # ── Layout: image + result side by side ───────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img_rgb, caption=uploaded.name, use_container_width=True)

    with col2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if status == "INTACT":
            st.markdown(f"""
            <div class="result-intact">
              <div class="result-title" style="color:#22c55e">✔ INTACT</div>
              <p class="result-sub">Hash matches registry exactly.<br>
              This image has NOT been tampered with.</p>
            </div>
            """, unsafe_allow_html=True)

        elif status == "TAMPERED":
            st.markdown(f"""
            <div class="result-tampered">
              <div class="result-title" style="color:#ef4444">✗ TAMPERED</div>
              <p class="result-sub">Hash mismatch detected.<br>
              This image has been modified since watermarking.</p>
            </div>
            """, unsafe_allow_html=True)

        else:  # UNKNOWN
            st.markdown(f"""
            <div class="result-unknown">
              <div class="result-title" style="color:#eab308">⚠ UNKNOWN</div>
              <p class="result-sub">Image not found in registry.<br>
              May not be a watermarked image from this dataset.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Hash details ──────────────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if result["image_id"]:
            st.markdown(f"<div class='section-title'>Image ID</div>"
                        f"<div class='hash-box'>{result['image_id']}</div>",
                        unsafe_allow_html=True)

        st.markdown(f"<div class='section-title' style='margin-top:10px'>Current hash (SHA-256)</div>"
                    f"<div class='hash-box'>{result['current_hash']}</div>",
                    unsafe_allow_html=True)

        if result["stored_hash"]:
            match_color = "#22c55e" if status == "INTACT" else "#ef4444"
            st.markdown(
                f"<div class='section-title' style='margin-top:10px'>Stored hash (SHA-256)</div>"
                f"<div class='hash-box' style='border-color:{match_color}'>"
                f"{result['stored_hash']}</div>",
                unsafe_allow_html=True,
            )

    # ── Hash diff visualizer (only for TAMPERED) ──────────────────────────────
    if status == "TAMPERED" and result["stored_hash"]:
        st.markdown("---")
        st.markdown("#### Hash difference")
        curr = result["current_hash"]
        stored = result["stored_hash"]
        diff_html = ""
        for a, b in zip(curr, stored):
            color = "#ef4444" if a != b else "#555"
            diff_html += f"<span style='color:{color}'>{a}</span>"
        st.markdown(
            f"<div class='hash-box' style='font-size:13px;line-height:2'>{diff_html}</div>"
            f"<div style='font-size:12px;color:#666;margin-top:6px'>"
            f"Red characters = positions where hash differs from stored value</div>",
            unsafe_allow_html=True,
        )

    # ── Pixel difference (if TAMPERED) ────────────────────────────────────────
    if status == "TAMPERED":
        st.markdown("---")
        with st.expander("Show pixel analysis"):
            st.info(
                "To see the exact pixel differences, upload both the original "
                "watermarked image and the suspected tampered image below."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                orig_file = st.file_uploader("Original watermarked image",
                                              type=["png","jpg","jpeg"],
                                              key="orig_diff")
            with col_b:
                tamp_file = st.file_uploader("Suspected tampered image",
                                              type=["png","jpg","jpeg"],
                                              key="tamp_diff")

            if orig_file and tamp_file:
                orig_bgr = cv2.imdecode(
                    np.frombuffer(orig_file.read(), dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                tamp_bgr = cv2.imdecode(
                    np.frombuffer(tamp_file.read(), dtype=np.uint8),
                    cv2.IMREAD_COLOR)

                if orig_bgr is not None and tamp_bgr is not None:
                    # Resize to same shape if needed
                    if orig_bgr.shape != tamp_bgr.shape:
                        tamp_bgr = cv2.resize(tamp_bgr,
                                               (orig_bgr.shape[1], orig_bgr.shape[0]))
                    diff = cv2.absdiff(orig_bgr, tamp_bgr)
                    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
                    diff_rgb = cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB)

                    c1, c2, c3 = st.columns(3)
                    c1.image(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB),
                             caption="Original", use_container_width=True)
                    c2.image(cv2.cvtColor(tamp_bgr, cv2.COLOR_BGR2RGB),
                             caption="Uploaded (suspected tampered)", use_container_width=True)
                    c3.image(diff_rgb, caption="Pixel diff ×5",
                             use_container_width=True)

                    changed_px = np.count_nonzero(diff.sum(axis=2))
                    total_px   = orig_bgr.shape[0] * orig_bgr.shape[1]
                    pct        = changed_px / total_px * 100
                    st.metric("Changed pixels",
                              f"{changed_px:,} / {total_px:,}",
                              delta=f"{pct:.2f}% of image")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER  — registry management helper
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("How to save your hash registry from Colab"):
    st.markdown("""
After running your pipeline in Colab, add this one cell to export the registry:

```python
import json

# Save hash_registry to Drive so the Streamlit app can load it
registry_path = "/content/drive/MyDrive/PCOS/hash_registry.json"
with open(registry_path, "w") as f:
    json.dump(hash_registry, f, indent=2)

print(f"Saved {len(hash_registry)} hashes to {registry_path}")
```

Then download `hash_registry.json` from Drive and place it in the same folder as `app.py`.
The app will load it automatically — no need to scan the image folder every time.
    """)