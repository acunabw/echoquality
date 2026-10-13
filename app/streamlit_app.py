from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from echoquality.application import (
    CaseFrame,
    CaseInput,
    analyze_complete_case,
    explore_single_frame,
    load_case_from_manifest,
    load_runtime,
    runtime_status,
    save_case_analysis,
)
from echoquality.config import load_config
from echoquality.data.io import load_image_with_metadata, load_mask, robust_scale

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="EchoQuality 2.0 Localhost",
    page_icon="\U0001fac0",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.eq-title {font-size: 2.1rem; font-weight: 750; color: #102A43; margin-bottom: 0.15rem;}
.eq-subtitle {color: #486581; margin-bottom: 1rem;}
.eq-warning {background: #FFF7E6; border: 1px solid #F6C453; padding: 0.75rem; border-radius: 0.5rem;}
.eq-card {border: 1px solid #D9E2EC; border-radius: 0.7rem; padding: 0.8rem; background: #FFFFFF;}
.small-note {font-size: 0.85rem; color: #627D98;}
</style>
""",
    unsafe_allow_html=True,
)


def _artifact_signature(root: Path, config_path: Path) -> tuple[float, ...]:
    candidates = [
        root / "quality_gate" / "hierarchical_quality_gate.joblib",
        root / "references" / "good_reference_view.joblib",
        root / "references" / "good_reference_frame.joblib",
        config_path,
    ]
    return tuple(path.stat().st_mtime if path.exists() else -1.0 for path in candidates)


@st.cache_resource(show_spinner=False)
def _cached_runtime(root_text: str, config_text: str, signature: tuple[float, ...]):
    del signature
    return load_runtime(root_text, config_path=config_text)


@st.cache_data(show_spinner=False)
def _cached_manifest(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path_text)


def _materialize_upload(uploaded: Any, folder: Path, fallback_name: str) -> Path:
    name = Path(getattr(uploaded, "name", fallback_name)).name
    path = folder / name
    path.write_bytes(uploaded.getvalue())
    return path


def _frame_from_upload(image_upload: Any, mask_upload: Any | None, phase: str, folder: Path) -> CaseFrame:
    image_path = _materialize_upload(image_upload, folder, f"{phase}_image.png")
    mask_path = _materialize_upload(mask_upload, folder, f"{phase}_mask.png") if mask_upload else None
    loaded = load_image_with_metadata(image_path)
    mask = load_mask(mask_path) if mask_path else None
    if mask is not None and mask.shape != loaded.array.shape:
        raise ValueError(f"La mascara {phase} tiene forma {mask.shape} y la imagen {loaded.array.shape}.")
    return CaseFrame(
        phase=phase,
        image=loaded.array,
        mask=mask,
        metadata=loaded.metadata,
        image_name=image_path.name,
        mask_name=mask_path.name if mask_path else "",
    )


def _overlay(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    base = robust_scale(image)
    rgb = np.repeat(base[..., None], 3, axis=2)
    if mask is None:
        return rgb
    labels = np.asarray(mask)
    colors = {
        1: np.array([0.95, 0.20, 0.20]),
        2: np.array([0.15, 0.75, 0.30]),
        3: np.array([0.15, 0.45, 0.95]),
    }
    for label, color in colors.items():
        region = labels == label
        rgb[region] = 0.45 * rgb[region] + 0.55 * color
    return np.clip(rgb, 0.0, 1.0)


def _result_zip(result: Any) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(result.summary, indent=2, ensure_ascii=False, default=str))
        archive.writestr("frame_features.csv", result.frame_features.to_csv(index=False))
        if not result.view_features.empty:
            archive.writestr("view_features.csv", result.view_features.to_csv(index=False))
        if not result.prediction.empty:
            archive.writestr("prediction.csv", result.prediction.to_csv(index=False))
        archive.writestr("level_summary.csv", result.level_summary.to_csv(index=False))
        archive.writestr("deviations.csv", result.deviations.to_csv(index=False))
        for phase, processing in result.processing.items():
            archive.writestr(f"processing_{phase}/candidate_scores.csv", processing.candidates.to_csv(index=False))
            image = Image.fromarray(np.round(robust_scale(processing.best_image) * 255).astype(np.uint8), mode="L")
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            archive.writestr(f"processing_{phase}/recommended.png", image_buffer.getvalue())
    return buffer.getvalue()


def _render_result(result: Any) -> None:
    st.markdown("### Resultado del caso")
    if result.mode == "complete":
        summary = result.summary
        metrics = st.columns(5)
        metrics[0].metric("Estado", str(summary["status"]))
        metrics[1].metric("P(NO APTA)", f"{100 * float(summary['probability_not_apt']):.1f}%")
        metrics[2].metric("Global", f"{100 * (1 - float(summary['score_global'])):.1f}%")
        metrics[3].metric("Anatomia", f"{100 * (1 - float(summary['score_anatomical'])):.1f}%")
        metrics[4].metric("Textura", f"{100 * (1 - float(summary['score_textural'])):.1f}%")
        st.caption(
            "Los porcentajes Global, Anatomia y Textura representan 1 - P(NO APTA) de cada modelo de nivel; "
            "son salidas del prototipo y no probabilidades clinicas."
        )
        info_col, ref_col = st.columns([0.9, 1.4])
        with info_col:
            st.markdown("#### Diagnostico jerarquico")
            st.json(
                {
                    "nivel_dominante": summary.get("dominant_level"),
                    "problema_dominante": summary.get("dominant_issue"),
                    "familias_de_procesamiento": summary.get("processing_families"),
                    "razon_de_decision": summary.get("decision_reason"),
                    "umbral": summary.get("threshold"),
                },
                expanded=True,
            )
        with ref_col:
            st.markdown("#### Distancia al patron Good")
            show = result.level_summary.copy()
            show["adequacy_score"] = show["adequacy_score"].map(lambda value: f"{100 * value:.1f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info(result.summary["message"])
        st.dataframe(result.level_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Imagenes y mascaras")
    phases = list(result.source_frames)
    columns = st.columns(max(1, len(phases)))
    for column, phase in zip(columns, phases):
        frame = result.source_frames[phase]
        with column:
            st.markdown(f"**{phase} - original + mascara**")
            st.image(_overlay(frame.image, frame.mask), use_container_width=True)
            if phase in result.processing:
                processing = result.processing[phase]
                st.markdown(f"**Recomendacion: {processing.recommendation}**")
                st.image(robust_scale(processing.best_image), clamp=True, use_container_width=True)
                st.caption(processing.best_spec.label())

    st.markdown("#### Desviaciones principales")
    if result.deviations.empty:
        st.info("No se generaron desviaciones.")
    else:
        st.dataframe(result.deviations, use_container_width=True, hide_index=True)

    if result.processing:
        st.markdown("#### Candidatos de procesamiento")
        for phase, processing in result.processing.items():
            with st.expander(f"{phase}: {processing.recommendation}", expanded=False):
                columns = [
                    column
                    for column in (
                        "candidate",
                        "target_level",
                        "valid",
                        "objective",
                        "ssim",
                        "edge_f1",
                        "anatomical_drift",
                        "texture_drift",
                        "clipping_fraction",
                    )
                    if column in processing.candidates
                ]
                st.dataframe(processing.candidates[columns].head(15), use_container_width=True, hide_index=True)

    download_col, save_col = st.columns(2)
    with download_col:
        st.download_button(
            "Descargar resultado del caso (.zip)",
            data=_result_zip(result),
            file_name=f"echoquality_{result.patient_id}_{result.view}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with save_col:
        output_root = st.text_input("Carpeta local para guardar", "results/localhost_cases", key="case_output_root")
        if st.button("Guardar resultado en disco", use_container_width=True):
            path = save_case_analysis(result, PROJECT_ROOT / output_root)
            st.success(f"Guardado en: {path}")


st.markdown('<div class="eq-title">EchoQuality 2.0 - Localhost</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="eq-subtitle">Analisis jerarquico de calidad ecocardiografica: Global + Anatomia + Textura + Fusion tardia + Good Reference + procesamiento adaptativo.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="eq-warning"><b>Prototipo academico.</b> No es un dispositivo medico y no debe utilizarse para decisiones clinicas.</div>',
    unsafe_allow_html=True,
)

st.sidebar.header("Configuracion del runtime")
experiment_root_text = st.sidebar.text_input("Carpeta del experimento entrenado", "results/camus500")
config_path_text = st.sidebar.text_input("YAML o configuracion resuelta", "configs/experiments/camus500.yaml")
experiment_root = (PROJECT_ROOT / experiment_root_text).resolve() if not Path(experiment_root_text).is_absolute() else Path(experiment_root_text)
config_path = (PROJECT_ROOT / config_path_text).resolve() if not Path(config_path_text).is_absolute() else Path(config_path_text)
status_table = runtime_status(experiment_root, config_path)
st.sidebar.dataframe(status_table, hide_index=True, use_container_width=True)
if st.sidebar.button("Recargar modelos y referencias", use_container_width=True):
    _cached_runtime.clear()
    st.rerun()

runtime = None
runtime_error = None
try:
    signature = _artifact_signature(experiment_root, config_path)
    runtime = _cached_runtime(str(experiment_root), str(config_path), signature)
except Exception as exc:
    runtime_error = exc

analysis_tab, orchestrator_tab, results_tab, methods_tab = st.tabs(
    ["Analizar caso", "Orquestador", "Resultados del experimento", "Metodo y limites"]
)

with analysis_tab:
    if runtime is None:
        st.warning(
            "No se encontraron los modelos y Good References requeridos. Primero ejecuta un experimento "
            "desde la pestana Orquestador."
        )
        st.code(str(runtime_error), language="text")
    else:
        st.success(f"Runtime cargado desde {experiment_root}")
        mode = st.radio(
            "Fuente del caso",
            [
                "Seleccionar desde la base CAMUS",
                "Cargar ED y ES con mascaras",
                "Cargar una sola imagen (exploratorio)",
            ],
            horizontal=True,
        )
        run_processing = st.checkbox("Ejecutar optimizacion adaptativa cuando corresponda", value=True)
        force_processing = st.checkbox(
            "Forzar evaluacion de candidatos aunque el estado no sea APTA_CONDICIONADA",
            value=False,
            help="Util para investigacion. El candidato identity siempre se incluye como control.",
        )

        if mode == "Seleccionar desde la base CAMUS":
            manifest_default = experiment_root / "manifest.csv"
            manifest_text = st.text_input("Manifest del experimento", str(manifest_default))
            manifest_path = Path(manifest_text)
            if not manifest_path.is_absolute():
                manifest_path = (PROJECT_ROOT / manifest_path).resolve()
            if not manifest_path.exists():
                st.error(f"No existe el manifest: {manifest_path}")
            else:
                manifest = _cached_manifest(str(manifest_path), manifest_path.stat().st_mtime)
                patient = st.selectbox("Paciente", sorted(manifest["patient_id"].astype(str).unique()))
                patient_rows = manifest[manifest["patient_id"].astype(str) == patient]
                view = st.selectbox("Vista", sorted(patient_rows["view"].astype(str).unique()))
                replace = st.checkbox("Reemplazar una fase de la base con una imagen cargada", value=False)
                replacement_phase = None
                replacement_image = None
                replacement_mask = None
                reuse_camus_mask = False
                if replace:
                    replacement_phase = st.selectbox("Fase que se reemplaza", ["ED", "ES"])
                    replacement_image = st.file_uploader(
                        "Imagen de reemplazo",
                        type=["png", "jpg", "jpeg", "tif", "tiff", "npy", "nii", "gz"],
                        key="replacement_image",
                    )
                    replacement_mask = st.file_uploader(
                        "Mascara de reemplazo",
                        type=["png", "tif", "tiff", "npy", "nii", "gz"],
                        key="replacement_mask",
                    )
                    reuse_camus_mask = st.checkbox(
                        "La imagen es el mismo frame CAMUS, solo procesado y geometricamente alineado; usar su mascara original",
                        value=False,
                        help=(
                            "Usa esta opcion solamente para comparar una transformacion del mismo frame. "
                            "Para una imagen diferente debes cargar su propia mascara."
                        ),
                    )
                if st.button("Ejecutar EchoQuality 2.0", type="primary", use_container_width=True):
                    try:
                        if replace and replacement_image is None:
                            raise ValueError("Seleccionaste reemplazar una fase, pero no cargaste la imagen de reemplazo.")
                        if replace and replacement_image is not None and replacement_mask is None and not reuse_camus_mask:
                            raise ValueError(
                                "Carga la mascara correspondiente o confirma que la imagen es una transformacion "
                                "alineada del mismo frame CAMUS."
                            )
                        with st.spinner("Extrayendo patrones, fusionando scores y comparando con Good Reference..."):
                            with tempfile.TemporaryDirectory(prefix="echoquality_upload_") as temp_text:
                                temp = Path(temp_text)
                                replacement_image_path = (
                                    _materialize_upload(replacement_image, temp, "replacement.png") if replacement_image else None
                                )
                                replacement_mask_path = (
                                    _materialize_upload(replacement_mask, temp, "replacement_mask.png") if replacement_mask else None
                                )
                                case = load_case_from_manifest(
                                    manifest_path,
                                    patient_id=patient,
                                    view=view,
                                    replacement_phase=replacement_phase,
                                    replacement_image_path=replacement_image_path,
                                    replacement_mask_path=replacement_mask_path,
                                )
                                result = analyze_complete_case(
                                    runtime,
                                    case,
                                    run_processing=run_processing,
                                    force_processing=force_processing,
                                )
                                st.session_state["eq_result"] = result
                    except Exception as exc:
                        st.exception(exc)

        elif mode == "Cargar ED y ES con mascaras":
            left, right = st.columns(2)
            with left:
                st.markdown("#### Fase ED")
                ed_image = st.file_uploader("Imagen ED", type=["png", "jpg", "jpeg", "tif", "tiff", "npy", "nii", "gz"], key="ed_image")
                ed_mask = st.file_uploader("Mascara ED", type=["png", "tif", "tiff", "npy", "nii", "gz"], key="ed_mask")
            with right:
                st.markdown("#### Fase ES")
                es_image = st.file_uploader("Imagen ES", type=["png", "jpg", "jpeg", "tif", "tiff", "npy", "nii", "gz"], key="es_image")
                es_mask = st.file_uploader("Mascara ES", type=["png", "tif", "tiff", "npy", "nii", "gz"], key="es_mask")
            patient_id = st.text_input("Identificador del caso", "uploaded_case")
            view = st.selectbox("Vista", ["2CH", "4CH"], key="upload_view")
            st.caption("Las mascaras deben usar etiquetas CAMUS: 1=LV, 2=MYO, 3=LA.")
            if st.button("Analizar archivos cargados", type="primary", use_container_width=True):
                if not all((ed_image, ed_mask, es_image, es_mask)):
                    st.error("Para el algoritmo completo debes cargar ED, ES y sus dos mascaras.")
                else:
                    try:
                        with st.spinner("Ejecutando EchoQuality 2.0..."):
                            with tempfile.TemporaryDirectory(prefix="echoquality_upload_") as temp_text:
                                temp = Path(temp_text)
                                case = CaseInput(
                                    patient_id=patient_id,
                                    view=view,
                                    source="uploaded_pair",
                                    frames={
                                        "ED": _frame_from_upload(ed_image, ed_mask, "ED", temp),
                                        "ES": _frame_from_upload(es_image, es_mask, "ES", temp),
                                    },
                                )
                                result = analyze_complete_case(
                                    runtime,
                                    case,
                                    run_processing=run_processing,
                                    force_processing=force_processing,
                                )
                                st.session_state["eq_result"] = result
                    except Exception as exc:
                        st.exception(exc)

        else:
            image_upload = st.file_uploader(
                "Imagen",
                type=["png", "jpg", "jpeg", "tif", "tiff", "npy", "nii", "gz"],
                key="single_image",
            )
            mask_upload = st.file_uploader(
                "Mascara CAMUS (recomendada)",
                type=["png", "tif", "tiff", "npy", "nii", "gz"],
                key="single_mask",
            )
            view = st.selectbox("Vista", ["2CH", "4CH"], key="single_view")
            phase = st.selectbox("Fase", ["ED", "ES"], key="single_phase")
            st.info(
                "Una sola imagen permite extraer patrones y compararla con Good Reference de frame. "
                "El Quality Gate final entrenado a nivel paciente-vista requiere ED y ES."
            )
            if st.button("Ejecutar analisis exploratorio", type="primary", use_container_width=True):
                if image_upload is None:
                    st.error("Carga una imagen.")
                else:
                    try:
                        with tempfile.TemporaryDirectory(prefix="echoquality_upload_") as temp_text:
                            frame = _frame_from_upload(image_upload, mask_upload, phase, Path(temp_text))
                            st.session_state["eq_result"] = explore_single_frame(runtime, frame, view=view)
                    except Exception as exc:
                        st.exception(exc)

        if "eq_result" in st.session_state:
            st.divider()
            _render_result(st.session_state["eq_result"])

with orchestrator_tab:
    st.markdown("### Entrenar o actualizar el pipeline desde YAML")
    st.warning(
        "Un experimento CAMUS500 puede tardar considerablemente. Para verificar la instalacion, comienza con "
        "smoke.yaml o camus10.yaml con la opcion fast."
    )
    configs = sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml"))
    labels = [str(path.relative_to(PROJECT_ROOT)) for path in configs]
    default_index = labels.index("configs/experiments/smoke.yaml") if "configs/experiments/smoke.yaml" in labels else 0
    selected_label = st.selectbox("Configuracion", labels, index=default_index)
    selected_config = PROJECT_ROOT / selected_label
    st.code(selected_config.read_text(encoding="utf-8"), language="yaml")
    stage_options = ["manifest", "features", "indicators", "gate", "references", "diagnostics", "processing", "report"]
    stages = st.multiselect("Etapas", stage_options, default=stage_options)
    col_fast, col_resume, col_force = st.columns(3)
    fast = col_fast.checkbox("Fast (prueba de ingenieria)", value=True)
    resume = col_resume.checkbox("Reanudar / omitir completadas", value=True)
    force = col_force.checkbox("Forzar repeticion", value=False)
    if st.button("Ejecutar orquestador", type="primary", use_container_width=True):
        command = [sys.executable, "-m", "echoquality", "run", "--config", str(selected_config)]
        if stages:
            command.extend(["--stages", *stages])
        if fast:
            command.append("--fast")
        if not resume:
            command.append("--no-resume")
        if force:
            command.append("--force")
        with st.spinner("Ejecutando pipeline. No cierres esta pagina..."):
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        st.code((completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else ""), language="text")
        if completed.returncode == 0:
            st.success("Experimento completado. Recarga el runtime desde la barra lateral.")
        else:
            st.error(f"El orquestador finalizo con codigo {completed.returncode}.")

with results_tab:
    st.markdown("### Resultados del experimento entrenado")
    model_summary = experiment_root / "quality_gate" / "model_summary.csv"
    if model_summary.exists():
        st.dataframe(pd.read_csv(model_summary), use_container_width=True, hide_index=True)
    else:
        st.info("Aun no existe model_summary.csv para esta carpeta.")
    image_paths = [
        experiment_root / "quality_gate" / "roc_quality_gate.png",
        experiment_root / "quality_gate" / "confusion_quality_gate.png",
        experiment_root / "quality_gate" / "level_scores.png",
    ]
    available = [path for path in image_paths if path.exists()]
    if available:
        columns = st.columns(len(available))
        for column, path in zip(columns, available):
            with column:
                st.image(str(path), caption=path.stem, use_container_width=True)
    report_path = experiment_root / "reports" / "experiment_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))

with methods_tab:
    st.markdown("### Que ejecuta el localhost")
    st.markdown(
        """
1. Carga el modelo jerarquico y los patrones Good ya entrenados.
2. Obtiene ED y ES, junto con las mascaras LV/MYO/LA.
3. Extrae representaciones Global, Anatomica y Textural.
4. Agrega fases con los mismos modos usados durante entrenamiento.
5. Calcula Score G, Score M y Score T.
6. El meta-clasificador realiza la fusion tardia y calcula P(NO APTA).
7. Compara cada nivel con su Good Reference y genera diagnostico por nivel.
8. Si procede, evalua transformaciones condicionadas al diagnostico y conserva la original cuando ninguna mejora cumple las restricciones.
        """
    )
    st.markdown("### Requisitos metodologicos")
    st.markdown(
        """
- El Quality Gate completo actual opera a nivel paciente-vista: necesita ED y ES.
- Los niveles Anatomico y Textural requieren una mascara con etiquetas CAMUS.
- La app no entrena un modelo por cada imagen; usa artefactos generados por un experimento reproducible.
- La textura se mide sobre la imagen controlada/original, no necesariamente sobre la imagen realzada para segmentacion.
- Todos los resultados son de investigacion y deben validarse antes de cualquier uso clinico.
        """
    )
