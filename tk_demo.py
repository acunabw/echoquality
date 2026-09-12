from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from echo_quality_pipeline.features import extract_features  # noqa: E402
from echo_quality_pipeline.image_io import load_image, load_mask, robust_scale, save_grayscale  # noqa: E402
from echo_quality_pipeline.optimization import optimize_image  # noqa: E402
from echo_quality_pipeline.quality_gate import load_bundle, predict_quality_gate  # noqa: E402
from echo_quality_pipeline.reference import GoodReference  # noqa: E402


class EchoQualityDemo(tk.Tk):
    """Interfaz de escritorio pensada para ejecutarse desde Visual Studio."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Calidad ecocardiografica - Pipeline de dos etapas")
        self.geometry("1240x780")
        self.minsize(1050, 700)
        self.image_path: Path | None = None
        self.mask_path: Path | None = None
        self.original: np.ndarray | None = None
        self.processed: np.ndarray | None = None
        self._photo_original: ImageTk.PhotoImage | None = None
        self._photo_processed: ImageTk.PhotoImage | None = None
        self.model_path = PROJECT_ROOT / "artifacts" / "quality_gate.joblib"
        self.reference_path = PROJECT_ROOT / "artifacts" / "good_reference.joblib"
        self._build_ui()
        self._load_default_example()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=12)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Sistema jerarquico: APTA/NO APTA + diagnostico y procesamiento adaptativo",
            font=("Segoe UI", 17, "bold"),
        ).pack(side="left")

        controls = ttk.Frame(self, padding=(12, 0, 12, 8))
        controls.pack(fill="x")
        ttk.Button(controls, text="Cargar imagen", command=self._choose_image).pack(side="left", padx=4)
        ttk.Button(controls, text="Cargar mascara (opcional)", command=self._choose_mask).pack(side="left", padx=4)
        ttk.Button(controls, text="Analizar y optimizar", command=self._analyze).pack(side="left", padx=12)
        ttk.Button(controls, text="Guardar imagen recomendada", command=self._save_processed).pack(side="left", padx=4)
        self.file_label = ttk.Label(controls, text="Sin archivo")
        self.file_label.pack(side="left", padx=12)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        image_panel = ttk.Frame(body)
        body.add(image_panel, weight=3)
        image_grid = ttk.Frame(image_panel)
        image_grid.pack(fill="both", expand=True)
        left = ttk.LabelFrame(image_grid, text="Imagen original", padding=6)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = ttk.LabelFrame(image_grid, text="Imagen recomendada", padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.original_label = ttk.Label(left, anchor="center")
        self.original_label.pack(fill="both", expand=True)
        self.processed_label = ttk.Label(right, anchor="center")
        self.processed_label.pack(fill="both", expand=True)

        info_panel = ttk.Frame(body, padding=(10, 0, 0, 0))
        body.add(info_panel, weight=2)
        self.status_label = tk.Label(
            info_panel,
            text="ESTADO: pendiente",
            font=("Segoe UI", 15, "bold"),
            bg="#E5E7EB",
            padx=10,
            pady=10,
        )
        self.status_label.pack(fill="x", pady=(0, 8))
        self.prob_label = ttk.Label(info_panel, text="Probabilidad NO APTA: -", font=("Segoe UI", 11))
        self.prob_label.pack(anchor="w", pady=4)
        self.recommend_label = ttk.Label(info_panel, text="Recomendacion: -", wraplength=420, font=("Segoe UI", 11, "bold"))
        self.recommend_label.pack(anchor="w", pady=4)

        ttk.Label(info_panel, text="Indicadores mas desviados del patron Good", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 3))
        self.indicator_tree = ttk.Treeview(info_panel, columns=("feature", "deficiency", "z"), show="headings", height=8)
        self.indicator_tree.heading("feature", text="Indicador")
        self.indicator_tree.heading("deficiency", text="Desviacion")
        self.indicator_tree.heading("z", text="z robusto")
        self.indicator_tree.column("feature", width=220)
        self.indicator_tree.column("deficiency", width=85, anchor="center")
        self.indicator_tree.column("z", width=80, anchor="center")
        self.indicator_tree.pack(fill="x")

        ttk.Label(info_panel, text="Mejores transformaciones validas", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 3))
        self.candidate_tree = ttk.Treeview(info_panel, columns=("candidate", "score", "ssim", "drift"), show="headings", height=8)
        self.candidate_tree.heading("candidate", text="Candidato")
        self.candidate_tree.heading("score", text="Objetivo")
        self.candidate_tree.heading("ssim", text="SSIM")
        self.candidate_tree.heading("drift", text="Drift textura")
        self.candidate_tree.column("candidate", width=235)
        self.candidate_tree.column("score", width=70, anchor="center")
        self.candidate_tree.column("ssim", width=60, anchor="center")
        self.candidate_tree.column("drift", width=85, anchor="center")
        self.candidate_tree.pack(fill="x")

        footer = ttk.Label(
            self,
            text="Prototipo academico. Dice del demo usa un segmentador proxy; en el estudio real debe conectarse el segmentador validado de la tesis.",
            anchor="center",
            padding=8,
        )
        footer.pack(fill="x")

    def _load_default_example(self) -> None:
        manifest_path = PROJECT_ROOT / "data" / "demo" / "manifest.csv"
        if not manifest_path.exists():
            return
        manifest = pd.read_csv(manifest_path)
        candidates = manifest[(manifest["quality"] == "Medium") & (manifest["phase"] == "ED")]
        row = candidates.iloc[0] if len(candidates) else manifest.iloc[0]
        self.image_path = Path(row["image_path"])
        self.mask_path = Path(row["mask_path"])
        self.file_label.configure(text=self.image_path.name)
        self.original = load_image(self.image_path)
        self._display(self.original, self.original_label, original=True)

    def _choose_image(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.tif *.tiff *.npy *.mhd *.nii *.nii.gz"), ("Todos", "*.*")])
        if filename:
            self.image_path = Path(filename)
            self.original = load_image(self.image_path)
            self.file_label.configure(text=self.image_path.name)
            self._display(self.original, self.original_label, original=True)

    def _choose_mask(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Mascaras", "*.png *.tif *.npy *.mhd *.nii *.nii.gz"), ("Todos", "*.*")])
        if filename:
            self.mask_path = Path(filename)

    def _display(self, array: np.ndarray, label: ttk.Label, *, original: bool) -> None:
        image = Image.fromarray(np.round(robust_scale(array) * 255).astype(np.uint8), mode="L")
        image.thumbnail((460, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        if original:
            self._photo_original = photo
        else:
            self._photo_processed = photo

    def _analyze(self) -> None:
        if self.original is None:
            messagebox.showwarning("Falta imagen", "Seleccione una imagen")
            return
        if not self.model_path.exists() or not self.reference_path.exists():
            messagebox.showerror(
                "Faltan artefactos",
                "Ejecute primero: python scripts/09_run_all_demo.py",
            )
            return
        mask = load_mask(self.mask_path) if self.mask_path and self.mask_path.exists() else None
        features = extract_features(self.original, mask)
        gate = predict_quality_gate(load_bundle(self.model_path), features)
        reference = GoodReference.load(self.reference_path)
        result = optimize_image(self.original, mask, reference, gate_status=gate["status"])
        self.processed = result.best_image
        self._display(self.processed, self.processed_label, original=False)

        color = {"APTA_DIRECTA": "#CDECCF", "APTA_CONDICIONADA": "#FFF0C7", "NO_APTA": "#F7C9C3"}[gate["status"]]
        self.status_label.configure(text=f"ESTADO: {gate['status']}", bg=color)
        self.prob_label.configure(text=f"Probabilidad NO APTA: {gate['proba_not_apt']:.3f} | umbral: {gate['threshold']:.3f}")
        self.recommend_label.configure(text=f"Recomendacion: {result.recommendation}\nProblemas dominantes: {', '.join(result.diagnostic.top_issues) or 'ninguno'}")

        for tree in (self.indicator_tree, self.candidate_tree):
            for item in tree.get_children():
                tree.delete(item)
        for _, row in result.diagnostic.indicator_table.head(8).iterrows():
            self.indicator_tree.insert("", "end", values=(row["feature"], f"{row['deficiency']:.2f}", f"{row['robust_z']:.2f}"))
        valid = result.candidates[result.candidates["valid"]].head(8)
        for _, row in valid.iterrows():
            self.candidate_tree.insert(
                "", "end",
                values=(row["candidate"], f"{row['objective']:.3f}", f"{row.get('ssim_original', 1.0):.3f}", f"{row.get('texture_drift', 0.0):.3f}"),
            )

    def _save_processed(self) -> None:
        if self.processed is None:
            messagebox.showwarning("Sin resultado", "Ejecute el analisis antes de guardar")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if filename:
            save_grayscale(filename, self.processed)
            messagebox.showinfo("Guardado", filename)


if __name__ == "__main__":
    EchoQualityDemo().mainloop()
