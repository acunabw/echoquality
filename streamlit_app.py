from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Instale Streamlit: pip install streamlit") from exc

from echo_quality_pipeline.features import extract_features
from echo_quality_pipeline.image_io import robust_scale
from echo_quality_pipeline.optimization import optimize_image
from echo_quality_pipeline.quality_gate import load_bundle, predict_quality_gate
from echo_quality_pipeline.reference import GoodReference


st.set_page_config(page_title="Calidad ecocardiografica", layout="wide")
st.title("Pipeline de dos etapas para calidad ecocardiografica")
st.caption("Compuerta APTA/NO APTA + diagnostico de indicadores + procesamiento adaptativo")

model_path = PROJECT_ROOT / "artifacts" / "quality_gate.joblib"
reference_path = PROJECT_ROOT / "artifacts" / "good_reference.joblib"
if not model_path.exists() or not reference_path.exists():
    st.error("Ejecute primero python scripts/09_run_all_demo.py")
    st.stop()

uploaded = st.file_uploader("Cargue una imagen PNG/JPG", type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded is None:
    manifest_path = PROJECT_ROOT / "data" / "demo" / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        row = manifest[manifest["quality"] == "Medium"].iloc[0]
        image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=float)
        mask = np.asarray(Image.open(row["mask_path"]), dtype=np.uint8)
        st.info("Se usa un ejemplo sintetico Medium incluido en el paquete.")
    else:
        st.stop()
else:
    image = np.asarray(Image.open(io.BytesIO(uploaded.getvalue())).convert("L"), dtype=float)
    mask = None

features = extract_features(image, mask)
gate = predict_quality_gate(load_bundle(model_path), features)
reference = GoodReference.load(reference_path)
result = optimize_image(image, mask, reference, gate_status=gate["status"])

c1, c2 = st.columns(2)
with c1:
    st.image(robust_scale(image), caption="Original", clamp=True)
with c2:
    st.image(result.best_image, caption="Recomendada", clamp=True)

m1, m2, m3 = st.columns(3)
m1.metric("Estado", gate["status"])
m2.metric("P(NO APTA)", f"{gate['proba_not_apt']:.3f}")
m3.metric("Transformacion", result.best_spec.name)
st.subheader("Indicadores alejados del patron Good")
st.dataframe(result.diagnostic.indicator_table.head(12), use_container_width=True)
st.subheader("Transformaciones candidatas")
st.dataframe(result.candidates.head(15), use_container_width=True)
st.warning("Prototipo academico. La salida no constituye una recomendacion clinica.")
