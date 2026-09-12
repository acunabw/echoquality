from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, text, face, edge="#173B57", fontsize=11.5, weight="bold"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, wrap=True)
    return patch


def arrow(ax, start, end, color="#284B63", style="-|>", connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16, linewidth=1.8, color=color, connectionstyle=connection))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/figures/pipeline_two_stage.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.955, "Pipeline de dos etapas: calidad, diagnostico y procesamiento adaptativo", ha="center", va="center", fontsize=22, weight="bold")

    # Paneles de etapa.
    panel1 = FancyBboxPatch((0.035, 0.18), 0.42, 0.69, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#F7FBFD", edgecolor="#6AAE8B", linewidth=2.1)
    panel2 = FancyBboxPatch((0.545, 0.18), 0.42, 0.69, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#FFFDF8", edgecolor="#E0A54B", linewidth=2.1)
    ax.add_patch(panel1)
    ax.add_patch(panel2)
    ax.text(0.245, 0.835, "ETAPA 1 - RECONOCIMIENTO DE PATRONES", ha="center", fontsize=15, weight="bold", color="#205E45")
    ax.text(0.755, 0.835, "ETAPA 2 - DIAGNOSTICO Y MEJORA CONTROLADA", ha="center", fontsize=15, weight="bold", color="#8A5400")

    # Etapa 1.
    add_box(ax, 0.075, 0.69, 0.34, 0.10, "Imagen B-mode + metadatos + mascara opcional", "#E8F1F8")
    add_box(ax, 0.075, 0.54, 0.34, 0.10, "Caracteristicas: intensidad, textura, nitidez, ruido y morfologia", "#D8EEF0", fontsize=11)
    add_box(ax, 0.075, 0.39, 0.34, 0.10, "SRP: probabilidad de NO APTA\nvalidacion anidada y agrupada por paciente", "#BFE3D0", fontsize=11)
    add_box(ax, 0.075, 0.235, 0.15, 0.095, "NO APTA\nrevisar o repetir", "#F7C9C3", fontsize=11)
    add_box(ax, 0.265, 0.235, 0.15, 0.095, "APTA / APTA\nCONDICIONADA", "#CDECCF", fontsize=11)
    arrow(ax, (0.245, 0.69), (0.245, 0.64))
    arrow(ax, (0.245, 0.54), (0.245, 0.49))
    arrow(ax, (0.19, 0.39), (0.15, 0.33), color="#9B2C2C", connection="arc3,rad=0.12")
    arrow(ax, (0.30, 0.39), (0.34, 0.33), color="#287A4D", connection="arc3,rad=-0.12")

    # Etapa 2.
    add_box(ax, 0.585, 0.69, 0.34, 0.10, "Patron Good: mediana, MAD, cuantiles y covarianza", "#E2D9F3", fontsize=11)
    add_box(ax, 0.585, 0.55, 0.34, 0.10, "Indicadores alejados: z robusto + distancia de Mahalanobis", "#FFF0C7", fontsize=11)
    add_box(ax, 0.585, 0.41, 0.34, 0.10, "Candidatos: CLAHE, gamma, median, bilateral, difusion, unsharp y kernels", "#FFE1B8", fontsize=10.8)
    add_box(ax, 0.585, 0.27, 0.34, 0.10, "Optimizar: calidad + contraste + ruido + Dice, sujeto a preservacion", "#FAD7B6", fontsize=10.8)
    add_box(ax, 0.585, 0.13, 0.34, 0.10, "Imagen recomendada + parametros + trazabilidad", "#CFE0F3", fontsize=11)
    arrow(ax, (0.755, 0.69), (0.755, 0.65))
    arrow(ax, (0.755, 0.55), (0.755, 0.51))
    arrow(ax, (0.755, 0.41), (0.755, 0.37))
    arrow(ax, (0.755, 0.27), (0.755, 0.23))

    # Conexion entre etapas y salida.
    arrow(ax, (0.415, 0.282), (0.585, 0.74), color="#287A4D", connection="arc3,rad=-0.22")
    ax.text(0.49, 0.54, "pasa\ncompuerta", ha="center", va="center", fontsize=10, color="#287A4D", weight="bold")
    arrow(ax, (0.15, 0.235), (0.50, 0.10), color="#9B2C2C", connection="arc3,rad=0.15")

    add_box(ax, 0.28, 0.035, 0.44, 0.085, "Salida para la tesis: ROI -> morfologia -> textura -> radiomics\nPreservar anatomia, escala y relaciones de intensidad", "#DCE6F2", fontsize=11.5)
    arrow(ax, (0.755, 0.13), (0.68, 0.12), connection="arc3,rad=0.10")
    ax.text(0.50, 0.005, "Principio: una imagen visualmente mas agradable no es necesariamente mejor para radiomics.", ha="center", fontsize=11.5, style="italic")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
