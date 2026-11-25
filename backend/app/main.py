from pathlib import Path
import io, os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import numpy as np

from .model import ModelService

BASE = Path(__file__).resolve().parents[1]
STATIC = BASE / "static"
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="NeuroMRI Research API",
    description="Bilingual research prototype. Not for clinical diagnosis.",
    version="0.1.0",
)
service = ModelService()
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")

@app.get("/api/status")
def status():
    return {
        "model_loaded": service.model is not None,
        "mode": "research_only",
        "message": "Experimental model loaded." if service.model else
                   "No trained model configured. Upload is available, but no prediction will be generated."
    }

def read_as_image(filename: str, content: bytes) -> Image.Image:
    ext = filename.lower()
    try:
        if ext.endswith((".png", ".jpg", ".jpeg")):
            return Image.open(io.BytesIO(content)).convert("L")
        if ext.endswith((".nii", ".nii.gz")):
            try:
                import nibabel as nib
            except ImportError:
                raise HTTPException(400, "NIfTI support requires: pip install nibabel")
            with tempfile.NamedTemporaryFile(suffix=".nii.gz" if ext.endswith(".nii.gz") else ".nii") as f:
                f.write(content); f.flush()
                vol = np.asarray(nib.load(f.name).dataobj)
            if vol.ndim == 4:
                vol = vol[..., 0]
            if vol.ndim != 3:
                raise HTTPException(400, "Expected a 3D NIfTI volume.")
            sl = vol[:, :, vol.shape[2] // 2]
            sl = np.nan_to_num(sl)
            lo, hi = np.percentile(sl, [1, 99])
            sl = np.clip((sl-lo) / (hi-lo + 1e-8), 0, 1) * 255
            return Image.fromarray(sl.astype("uint8")).convert("L")
        if ext.endswith(".dcm"):
            try:
                import pydicom
            except ImportError:
                raise HTTPException(400, "DICOM support requires: pip install pydicom")
            ds = pydicom.dcmread(io.BytesIO(content))
            arr = ds.pixel_array.astype("float32")
            arr = np.nan_to_num(arr)
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip((arr-lo)/(hi-lo+1e-8), 0, 1)*255
            return Image.fromarray(arr.astype("uint8")).convert("L")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {str(e)}")
    raise HTTPException(415, "Unsupported file. Use PNG, JPG, NIfTI, or DICOM.")

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    name = file.filename or "upload"
    allowed = (".png", ".jpg", ".jpeg", ".nii", ".nii.gz", ".dcm")
    if not name.lower().endswith(allowed):
        raise HTTPException(415, "Unsupported file type.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "File exceeds 25 MB limit.")
    image = read_as_image(name, content)
    # Do not save uploaded patient data to disk.
    result = service.predict(image)
    if result is None:
        return {
            "status": "model_unavailable",
            "message": "فایل با موفقیت خوانده شد، اما مدل آموزش‌دیده نصب نیست. هیچ نتیجه‌ای تولید نشده است.",
            "message_en": "The file was read successfully, but no trained model is configured. No result was generated."
        }
    return {
        "status": "experimental_output",
        "scores": result,
        "warning": "Research output only; not a diagnosis. Must not guide clinical decisions."
    }
