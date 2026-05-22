#!/usr/bin/env python3
"""
4_plots.py – Visualisations de méthylation CpG par cohorte
===========================================================
Prend en entrée les fichiers TSV de 3_cohort_analysis.py et génère des
visualisations Plotly interactives (HTML) organisées en trois niveaux :

  01_global/   – Vues d'ensemble du cohorte (4 plots)
  02_samples/  – Un sous-dossier par échantillon (3 plots)
  03_loci/     – Un sous-dossier par locus outlier (2 plots, top N loci)

Plots générés
-------------
  01_global/
    01_manhattan_outlier_count.html  – Manhattan : Y = nb d'outliers par locus
    02_outlier_scatter.html          – cohort_mean vs z_score (4 subplots haplotypes)
    03_global_stats.html             – distributions : taille, nb CpG, méthylation
    04_heatmap_top_outliers.html     – heatmap z_score top 100 loci outliers

  02_samples/<sample>/
    01_volcano.html                  – Δ méthylation vs -log10(p_adj), 4 subplots
    02_hap1_vs_hap2.html             – scatter hap1 vs hap2
    03_manhattan.html                – z_score par position génomique

  03_loci/<island>/
    01_boxplot_haplotypes.html       – distribution frac_meth, 4 subplots haplotypes
    02_hap1_vs_hap2.html             – hap1 vs hap2 par échantillon

Usage
-----
  python 4_plots.py \\
      --stats      results/02_stats/all_samples_stats.tsv \\
      --outdir     results/03_plots \\
      [--locus_stats  results/02_stats/cohort_locus_stats.tsv] \\
      [--outliers     results/02_stats/outliers.tsv]           \\
      [--top_loci     100]   \\
      [--z_thresh     2.5]   \\
      [--manhattan_hap all]  \\
      [--top_labels   3]     \\
      [--no_per_sample]      \\
      [--no_per_loci]

Dépendances : pip install plotly pandas numpy
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly requis : pip install plotly")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import umap as _umap_module
    _UMAP_OK = True
except ImportError:
    _UMAP_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
CHROM_SIZES_GRCH38: dict[str, int] = {
    "chr1":  248_956_422, "chr2":  242_193_529, "chr3":  198_295_559,
    "chr4":  190_214_555, "chr5":  181_538_259, "chr6":  170_805_979,
    "chr7":  159_345_973, "chr8":  145_138_636, "chr9":  138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16":  90_338_345, "chr17":  83_257_441, "chr18":  80_373_285,
    "chr19":  58_617_616, "chr20":  64_444_167, "chr21":  46_709_983,
    "chr22":  50_818_468, "chrX":  156_040_895, "chrY":   57_227_415,
}
CHROM_ORDER = list(CHROM_SIZES_GRCH38.keys())
_BAND_COLORS = ["rgba(210,210,210,0.25)", "rgba(245,245,245,0.0)"]
PLOT_BGCOLOR  = "#f7f7f7"   # fond de la zone de tracé  (modifiable ici)
PAPER_BGCOLOR = "#fffdfd"   # fond global (marges, titre)

HAP_ORDER  = ["hap1", "hap2", "unphased", "all"]
HAP_COLORS = {"hap1": "#1f77b4", "hap2": "#d62728", "unphased": "#7f7f7f", "all": "#2ca02c"}
HAP_TITLES = {
    "hap1": "Haplotype 1", "hap2": "Haplotype 2",
    "unphased": "Non phasé",  "all": "Tous reads",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(s: str) -> str:
    s = re.sub(r'[:\\/*?"<>|]', '_', str(s))
    return re.sub(r'_+', '_', s).strip('_')[:80]


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _cumulative_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    cumpos = 0
    for chrom in CHROM_ORDER:
        offsets[chrom] = cumpos
        cumpos += CHROM_SIZES_GRCH38[chrom]
    return offsets


def _manhattan_layout_elements(offsets: dict[str, int]) -> tuple:
    """Retourne (shapes, tick_x, tick_labels) pour l'axe X chromosomique."""
    shapes, ticks_x, ticks_t = [], [], []
    for i, chrom in enumerate(CHROM_ORDER):
        if chrom not in offsets:
            continue
        x0 = offsets[chrom]
        x1 = x0 + CHROM_SIZES_GRCH38[chrom]
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=x0, x1=x1, y0=0, y1=1,
            fillcolor=_BAND_COLORS[i % 2], line_width=0, layer="below",
        ))
        ticks_x.append((x0 + x1) / 2)
        ticks_t.append(chrom.replace("chr", ""))
    return shapes, ticks_x, ticks_t


def _write(fig: "go.Figure", path: str) -> None:
    if os.path.exists(path):
        log.info("    [SKIP] fichier déjà existant : %s", path)
        return
    fig.write_html(path, include_plotlyjs="cdn")
    log.info("    → %s", path)


def _pval_col(df: pd.DataFrame) -> str:
    """Retourne 'p_adj' si disponible, sinon 'p_value'."""
    return "p_adj" if "p_adj" in df.columns else "p_value"


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    stats_path: str,
    locus_path: str,
    outliers_path: str,
) -> tuple:
    log.info("Chargement des données …")
    if not os.path.isfile(stats_path):
        sys.exit(f"[ERROR] Fichier introuvable : {stats_path}")
    stats = pd.read_csv(stats_path, sep="\t")

    locus = pd.read_csv(locus_path, sep="\t") if os.path.isfile(locus_path) else pd.DataFrame()
    out_df = pd.read_csv(outliers_path, sep="\t") if os.path.isfile(outliers_path) else pd.DataFrame()

    # Normalisation des types booléens
    for col in ("IQR_outlier", "Z_outlier", "outlier"):
        if col in stats.columns:
            stats[col] = stats[col].astype(bool)
        if not out_df.empty and col in out_df.columns:
            out_df[col] = out_df[col].astype(bool)

    log.info("  all_samples_stats : %d lignes  (%d échantillons, %d loci)",
             len(stats), stats["sample"].nunique(), stats["island_id"].nunique())
    if not locus.empty:
        log.info("  cohort_locus_stats: %d lignes", len(locus))
    if not out_df.empty:
        log.info("  outliers          : %d lignes  (%d loci, %d échantillons)",
                 len(out_df), out_df["island_id"].nunique(), out_df["sample"].nunique())
    return stats, locus, out_df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS GLOBAUX
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_manhattan_outlier_count(
    outliers: pd.DataFrame,
    locus_stats: pd.DataFrame,
    offsets: dict[str, int],
    outdir: str,
) -> None:
    """
    Manhattan : un point par locus.
    Y     = nombre d'échantillons outliers (haplotype = 'all').
    Taille = log(n_outliers).
    Couleur = z_score moyen des outliers (rouge hyper, bleu hypo).
    """
    _out = os.path.join(outdir, "01_manhattan_outlier_count.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    if outliers.empty:
        log.warning("[Global] Pas d'outliers – manhattan outlier count ignoré.")
        return

    sub = outliers[outliers["haplotype"] == "all"] if "haplotype" in outliers.columns else outliers
    if sub.empty:
        sub = outliers  # fallback

    agg = sub.groupby("island_id").agg(
        n_outliers = ("sample",  "nunique"),
        mean_z     = ("z_score", "mean"),
    ).reset_index()

    # Positions depuis locus_stats (hap='all') ou directement depuis outliers
    if not locus_stats.empty and "island_id" in locus_stats.columns:
        pos_df = locus_stats[locus_stats["haplotype"] == "all"][[
            "island_id", "island_name", "chrom", "island_start", "island_end",
            "cohort_n_samples",
        ]].drop_duplicates("island_id")
    else:
        pos_df = sub[["island_id", "island_name", "chrom",
                      "island_start", "island_end"]].drop_duplicates("island_id")

    df = agg.merge(pos_df, on="island_id", how="left").dropna(subset=["chrom"])
    df = df[df["chrom"].isin(CHROM_ORDER)].copy()
    df["_ci"] = df["chrom"].map({c: i for i, c in enumerate(CHROM_ORDER)})
    df = df.sort_values(["_ci", "island_start"])

    mid = (df["island_start"] + df["island_end"]) // 2
    df["x_pos"] = df["chrom"].map(offsets) + mid

    cmax = float(max(df["mean_z"].abs().quantile(0.99), 1.0))
    marker_size = np.clip(4 + np.log1p(df["n_outliers"]) * 3, 4, 18)
    shapes, ticks_x, ticks_t = _manhattan_layout_elements(offsets)
    total_len = sum(CHROM_SIZES_GRCH38.values())

    n_samp_str = df.get("cohort_n_samples", pd.Series(["?"] * len(df))).fillna("?").astype(str)
    hover = (
        "<b>" + df["island_name"].fillna(df["island_id"]) + "</b><br>"
        + df["island_id"] + "<br>"
        + "Outliers : " + df["n_outliers"].astype(str) + " / " + n_samp_str + "<br>"
        + "Z moyen  : " + df["mean_z"].round(2).astype(str)
    )

    fig = go.Figure(go.Scatter(
        x=df["x_pos"], y=df["n_outliers"],
        mode="markers",
        marker=dict(
            color=df["mean_z"], colorscale="RdBu_r",
            cmin=-cmax, cmax=cmax,
            size=marker_size, opacity=0.80, line=dict(width=0),
            colorbar=dict(title="Z moyen<br>outliers", len=0.6, thickness=14, x=1.01),
            showscale=True,
        ),
        text=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title="Nombre d'échantillons outliers par locus CpG (haplotype = all)",
        xaxis=dict(
            title="Position chromosomique (GRCh38)",
            tickvals=ticks_x, ticktext=ticks_t,
            tickfont=dict(size=9), showgrid=False, zeroline=False,
            range=[-total_len * 0.005, total_len * 1.01],
        ),
        yaxis=dict(title="Nb outliers", gridcolor="rgba(200,200,200,0.4)",
                   zeroline=False),
        shapes=shapes,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        height=460, width=1700,
        margin=dict(l=60, r=80, t=60, b=50),
    )
    _write(fig, os.path.join(outdir, "01_manhattan_outlier_count.html"))


def plot_global_outlier_scatter(
    stats: pd.DataFrame,
    z_thresh: float,
    outdir: str,
) -> None:
    """
    Scatter : cohort_mean (X) vs z_score (Y), couleur = frac_meth.
    4 subplots : hap1, hap2, unphased, all.
    Outliers représentés avec des marqueurs plus grands et bordurés.
    """
    _out = os.path.join(outdir, "02_outlier_scatter.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    df = stats.dropna(subset=["z_score", "cohort_mean", "frac_meth"]).copy()
    if df.empty:
        log.warning("[Global] Données insuffisantes pour outlier scatter.")
        return

    # Ne conserver que les points déviés pour éviter la surcharge visuelle
    if "outlier" in df.columns:
        df = df[df["outlier"].fillna(False) | (df["z_score"].abs() >= z_thresh)]
    else:
        df = df[df["z_score"].abs() >= z_thresh]
    if df.empty:
        log.warning("[Global] Aucun point dévié (|Z|≥%.1f) – outlier scatter ignoré.", z_thresh)
        return

    # Limiter aux 10 000 valeurs les plus significatives (|z_score| décroissant)
    if len(df) > 10_000:
        df = df.assign(_abs_z=df["z_score"].abs()).nlargest(10_000, "_abs_z").drop(columns="_abs_z")
        log.info("  [Outlier scatter] Limité aux 10 000 valeurs les plus significatives.")

    haps_present = [h for h in HAP_ORDER if h in df["haplotype"].unique()]
    ncols = 2
    nrows = -(-len(haps_present) // ncols)

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[HAP_TITLES.get(h, h) for h in haps_present],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    for idx, hap in enumerate(haps_present):
        row = idx // ncols + 1
        col = idx % ncols + 1
        sub = df[df["haplotype"] == hap]
        is_out = sub.get("outlier", pd.Series(False, index=sub.index)).fillna(False)

        for flag, size, alpha, border_w in [(False, 4, 0.35, 0), (True, 8, 0.85, 1)]:
            pts = sub[is_out == flag]
            if pts.empty:
                continue
            first_trace = (idx == 0 and not flag)  # colorbar une seule fois
            fig.add_trace(go.Scatter(
                x=pts["cohort_mean"],
                y=pts["z_score"],
                mode="markers",
                marker=dict(
                    color=pts["frac_meth"],
                    colorscale="RdBu_r", cmin=0, cmax=1,
                    size=size, opacity=alpha,
                    line=dict(width=border_w, color="black"),
                    showscale=first_trace,
                    colorbar=dict(title="Méth.<br>sample", len=0.35,
                                  thickness=12, x=1.02) if first_trace else None,
                ),
                text=(
                    "<b>" + pts["island_name"].fillna(pts["island_id"]) + "</b><br>"
                    + pts["sample"] + "<br>"
                    + "Z: " + pts["z_score"].round(2).astype(str) + "<br>"
                    + "Meth sample  : " + pts["frac_meth"].round(3).astype(str) + "<br>"
                    + "Meth cohorte : " + pts["cohort_mean"].round(3).astype(str)
                    + ("<br>Méd. cohorte : " + pts["cohort_median"].round(3).astype(str)
                       if "cohort_median" in pts.columns else "")
                    + ("<br>ÉT cohorte   : " + pts["cohort_std"].round(3).astype(str)
                       if "cohort_std" in pts.columns else "")
                ),
                hoverinfo="text", showlegend=False,
            ), row=row, col=col)

        # Lignes de seuil
        for sign, color in [(1, "rgba(190,40,40,0.55)"), (-1, "rgba(30,80,200,0.55)")]:
            fig.add_hline(y=sign * z_thresh, line_dash="dash",
                          line_color=color, line_width=1, row=row, col=col)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.15)",
                      line_width=1, row=row, col=col)

    fig.update_layout(
        title=(
            "Scatter cohorte : méthylation moyenne vs Z-score  "
            f"(traits pointillés = |Z| ≥ {z_thresh})"
        ),
        height=440 * nrows, width=1200,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        margin=dict(l=60, r=90, t=80, b=50),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.layout[ax].update(title="Méth. moyenne cohorte", showgrid=False,
                                  range=[-0.02, 1.02])
        if ax.startswith("yaxis"):
            fig.layout[ax].update(title="Z-score",
                                  gridcolor="rgba(200,200,200,0.4)")
    _write(fig, os.path.join(outdir, "02_outlier_scatter.html"))


def plot_global_stats_distributions(
    locus_stats: pd.DataFrame,
    outdir: str,
) -> None:
    """
    Distributions : taille des îlots, nombre de CpG annotés, méthylation
    moyenne cohorte. 3 histogrammes en une figure.
    """
    _out = os.path.join(outdir, "03_global_stats.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    df = locus_stats[locus_stats["haplotype"] == "all"].copy() if not locus_stats.empty else pd.DataFrame()
    if df.empty:
        log.warning("[Global] Pas de données haplotype='all' pour stats distributions.")
        return

    df["island_size"] = df["island_end"] - df["island_start"]

    panels = [
        ("island_size",   "Taille des îlots (pb)",              "steelblue",  50),
        ("island_cpgNum", "Nombre de CpG annotés (UCSC)",        "darkorange", 40),
        ("cohort_mean",   "Méthylation moyenne cohorte",         "seagreen",   50),
    ]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.07,
    )

    for col, (var, xlabel, color, nbins) in enumerate(panels, start=1):
        if var not in df.columns:
            continue
        vals = df[var].dropna()
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=nbins, marker_color=color,
                         opacity=0.8, name=xlabel),
            row=1, col=col,
        )
        fig.update_xaxes(title_text=xlabel, row=1, col=col)
        fig.update_yaxes(title_text="Nb de loci" if col == 1 else "",
                         gridcolor="rgba(200,200,200,0.4)", row=1, col=col)

    n_loci   = int(df["island_id"].nunique()) if "island_id" in df.columns else len(df)
    n_samp   = int(df["cohort_n_samples"].median()) if "cohort_n_samples" in df.columns else "?"
    fig.update_layout(
        title=(
            f"Statistiques globales des îlots CpG – {n_loci} loci  "
            f"(médiane {n_samp} échantillons/locus)"
        ),
        height=420, width=1200,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=50),
    )
    _write(fig, os.path.join(outdir, "03_global_stats.html"))


def plot_global_heatmap_outliers(
    stats: pd.DataFrame,
    outdir: str,
    top_n: int = 100,
) -> None:
    """
    Heatmap z_score : lignes = top N loci outliers, colonnes = échantillons.
    Haplotype = 'all'. Lignes triées par z_score moyen (hypo en bas, hyper en haut).
    """
    _out = os.path.join(outdir, "04_heatmap_top_outliers.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    df = stats[stats["haplotype"] == "all"].copy()
    if df.empty or "z_score" not in df.columns:
        log.warning("[Global] Données insuffisantes pour heatmap outliers.")
        return

    # Sélection des top N loci par nombre d'outliers
    if "outlier" in df.columns:
        counts = (
            df[df["outlier"]]
            .groupby("island_id")["sample"].nunique()
        )
    else:
        counts = df.groupby("island_id")["z_score"].apply(
            lambda z: int((z.abs() >= 2.5).sum())
        )
    top_islands = counts.nlargest(top_n).index.tolist()

    if not top_islands:
        log.warning("[Global] Aucun locus outlier détecté – heatmap ignorée.")
        return

    sub = df[df["island_id"].isin(top_islands)]
    pivot = sub.pivot_table(index="island_id", columns="sample", values="z_score")

    # Tri par z_score moyen (hypométhylés en bas, hyperméthylés en haut)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    name_map = df.drop_duplicates("island_id").set_index("island_id")["island_name"]
    y_labels = [
        f"{name_map.get(iid, '')} | {iid}" for iid in pivot.index
    ]

    vals = pivot.values.ravel()
    vals_valid = vals[~np.isnan(vals)]
    cmax = float(max(vals_valid.std() * 3, 2.5)) if len(vals_valid) > 1 else 3.0

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=y_labels,
        colorscale="RdBu_r",
        zmid=0, zmin=-cmax, zmax=cmax,
        colorbar=dict(title="Z-score"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=(
            f"Heatmap Z-score – top {len(top_islands)} loci outliers "
            "(haplotype = all, rouge=hyper, bleu=hypo)"
        ),
        xaxis=dict(title="Échantillon", tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=8)),
        height=max(500, len(pivot) * 14),
        width=max(900, len(pivot.columns) * 22 + 250),
        margin=dict(l=220, r=80, t=70, b=110),
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
    )
    _write(fig, os.path.join(outdir, "04_heatmap_top_outliers.html"))


def plot_global_methylation_embedding(
    stats: pd.DataFrame,
    outdir: str,
) -> None:
    """
    Visualisation 2D des profils globaux de méthylation : 1 point par échantillon.
    Utilise les données haplotype='all' (tous reads).
    Méthode : UMAP si umap-learn est installé, sinon PCA (scikit-learn).
    Axe de couleur = méthylation moyenne de l'échantillon.
    """
    _out = os.path.join(outdir, "05_methylation_embedding.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    if not _SKLEARN_OK and not _UMAP_OK:
        log.warning(
            "[Global] sklearn et umap non disponibles – plot embedding ignoré.\n"
            "         Installer : pip install scikit-learn   ou   pip install umap-learn"
        )
        return

    df = stats[stats["haplotype"] == "all"].copy()
    if df.empty:
        log.warning("[Global] Pas de données haplotype='all' pour l'embedding.")
        return

    # Matrice échantillons × loci  (frac_meth)
    pivot = df.pivot_table(index="sample", columns="island_id", values="frac_meth")
    if pivot.shape[0] < 3:
        log.warning("[Global] Trop peu d'échantillons (%d) pour l'embedding.", pivot.shape[0])
        return

    # Imputation NaN par la médiane de chaque locus
    pivot_filled = pivot.fillna(pivot.median())
    X = pivot_filled.values
    samples = pivot_filled.index.tolist()

    # Statistiques par échantillon pour le hover
    sample_stats = df.groupby("sample").agg(
        mean_meth=("frac_meth", "mean"),
        n_loci=("island_id", "nunique"),
    )
    if "outlier" in df.columns:
        n_out = (
            df[df["outlier"].fillna(False)]
            .groupby("sample")["island_id"].nunique()
            .rename("n_outliers")
        )
        sample_stats = sample_stats.join(n_out, how="left").fillna({"n_outliers": 0})
        sample_stats["n_outliers"] = sample_stats["n_outliers"].astype(int)

    # ── Calcul de l'embedding ────────────────────────────────────────────────
    method_label = ""
    coords = None

    if _UMAP_OK:
        try:
            reducer = _umap_module.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(X)
            method_label = "UMAP"
            log.info("  Embedding UMAP calculé (%d échantillons × %d loci).", *X.shape)
        except Exception as exc:
            log.warning("  UMAP échoué (%s) – bascule sur PCA.", exc)

    if coords is None and _SKLEARN_OK:
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_ * 100
        method_label = f"PCA  (PC1 = {ev[0]:.1f} %,  PC2 = {ev[1]:.1f} %)"
        log.info("  Embedding PCA calculé (%d échantillons × %d loci).", *X.shape)

    if coords is None:
        log.warning("[Global] Impossible de calculer l'embedding.")
        return

    embed_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "sample": samples,
    }).merge(sample_stats.reset_index(), on="sample", how="left")

    hover = (
        "<b>" + embed_df["sample"] + "</b><br>"
        + "Méth. moyenne : " + embed_df["mean_meth"].round(3).astype(str) + "<br>"
        + "Loci couverts : " + embed_df["n_loci"].astype(int).astype(str)
        + ("<br>Outliers      : " + embed_df["n_outliers"].astype(str)
           if "n_outliers" in embed_df.columns else "")
    )

    is_umap = method_label.startswith("UMAP")
    ax1 = "UMAP1" if is_umap else "PC1"
    ax2 = "UMAP2" if is_umap else "PC2"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embed_df["x"],
        y=embed_df["y"],
        mode="markers",
        marker=dict(
            color=embed_df["mean_meth"],
            colorscale="RdBu_r",
            cmin=0, cmax=1,
            size=10, opacity=0.85,
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            colorbar=dict(title="Méth.<br>moyenne", thickness=14, len=0.6),
            showscale=True,
        ),
        text=hover,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=(
            f"Profils globaux de méthylation – {method_label}<br>"
            f"<sub>{pivot.shape[1]} loci · {len(samples)} échantillons · haplotype = all</sub>"
        ),
        xaxis=dict(title=ax1, showgrid=False, zeroline=False),
        yaxis=dict(title=ax2, showgrid=False, zeroline=False),
        height=650, width=850,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        showlegend=False,
        margin=dict(l=60, r=110, t=90, b=50),
    )
    _write(fig, os.path.join(outdir, "05_methylation_embedding.html"))


def plot_global_manhattan_outliers_by_haplotype(
    outliers: pd.DataFrame,
    stats: pd.DataFrame,
    offsets: dict[str, int],
    outdir: str,
    top_n: int = 10_000,
) -> None:
    """
    Manhattan 4 sous-graphes (un par haplotype) :
    Chaque point = un outlier individuel (sample × locus).
    Y = |z_score|. Cappé aux top_n outliers les plus significatifs par haplotype.
    Couleurs alternées par chromosome (style Manhattan classique).
    """
    _out = os.path.join(outdir, "06_manhattan_outliers_by_haplotype.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    if outliers.empty:
        log.warning("[Global] Pas d'outliers – manhattan outliers par haplotype ignoré.")
        return

    haps_present = [h for h in HAP_ORDER if h in outliers["haplotype"].unique()]
    if not haps_present:
        return

    ncols = 2
    nrows = -(-len(haps_present) // ncols)
    total_len = sum(CHROM_SIZES_GRCH38.values())
    _, ticks_x, ticks_t = _manhattan_layout_elements(offsets)

    # Paire de couleurs (saturée / pâle) par haplotype pour l'alternance chromosomique
    _CHROM_PALETTE = {
        "hap1":     ("#1f77b4", "#a8c8e8"),
        "hap2":     ("#d62728", "#f0a0a0"),
        "unphased": ("#7f7f7f", "#c0c0c0"),
        "all":      ("#2ca02c", "#98df8a"),
    }

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[HAP_TITLES.get(h, h) for h in haps_present],
        horizontal_spacing=0.06, vertical_spacing=0.15,
    )

    for idx, hap in enumerate(haps_present):
        row   = idx // ncols + 1
        col_s = idx %  ncols + 1

        sub = outliers[outliers["haplotype"] == hap].copy()
        if sub.empty:
            continue

        sub["abs_z"] = sub["z_score"].abs()

        # Cap aux top_n outliers les plus significatifs
        capped = False
        if len(sub) > top_n:
            sub = sub.nlargest(top_n, "abs_z")
            capped = True

        # outliers contient déjà island_name, chrom, island_start, island_end
        df = sub.dropna(subset=["chrom"])
        df = df[df["chrom"].isin(CHROM_ORDER)].copy()
        df["_ci"] = df["chrom"].map({c: i for i, c in enumerate(CHROM_ORDER)})
        df = df.sort_values(["_ci", "island_start"])
        mid = (df["island_start"] + df["island_end"]) // 2
        df["x_pos"] = df["chrom"].map(offsets) + mid

        n_total = len(outliers[outliers["haplotype"] == hap])
        subtitle = f" (top {top_n:,} / {n_total:,})" if capped else f" ({n_total:,} points)"
        log.info("  Haplotype %s : %d outliers affichés%s.",
                 hap, len(df), f" sur {n_total}" if capped else "")

        col_hi, col_lo = _CHROM_PALETTE.get(hap, ("#4682b4", "#9fc4e0"))
        for ci, chrom in enumerate(CHROM_ORDER):
            pts = df[df["chrom"] == chrom]
            if pts.empty:
                continue
            color = col_hi if ci % 2 == 0 else col_lo
            hover = (
                "<b>" + pts["island_name"].fillna(pts["island_id"]) + "</b><br>"
                + "Sample : " + pts["sample"].astype(str) + "<br>"
                + chrom + "<br>"
                + "frac_meth : " + pts["frac_meth"].round(3).astype(str)
                + "  (moy. cohorte : " + pts["cohort_mean"].round(3).astype(str) + ")<br>"
                + "Z-score : " + pts["z_score"].round(3).astype(str)
            )
            fig.add_trace(go.Scatter(
                x=pts["x_pos"], y=pts["z_score"],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.65, line=dict(width=0)),
                text=hover, hoverinfo="text", showlegend=False,
            ), row=row, col=col_s)

        # Annotation du nombre de points dans le sous-graphe
        fig.add_annotation(
            text=subtitle,
            xref="paper", yref="paper",
            x=(col_s - 0.5) / ncols, y=1.0 - (row - 1) / nrows,
            xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(size=9, color="#555"),
        )

        fig.update_xaxes(
            tickvals=ticks_x, ticktext=ticks_t,
            tickfont=dict(size=8), showgrid=False, zeroline=False,
            range=[-total_len * 0.005, total_len * 1.01],
            row=row, col=col_s,
        )
        fig.update_yaxes(
            title_text="Z-score" if col_s == 1 else "",
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=True, zerolinecolor="rgba(100,100,100,0.5)", zerolinewidth=1,
            row=row, col=col_s,
        )

    fig.update_layout(
        title=(
            "Outliers individuels par haplotype – Manhattan bilatéral (4 sous-graphes)<br>"
            f"<sub>Un point = un échantillon outlier à un locus  ·  Y = Z-score signé  ·  "
            f"top {top_n:,} par |Z| si > {top_n:,} points</sub>"
        ),
        height=380 * nrows, width=1700,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        margin=dict(l=60, r=40, t=90, b=50),
    )
    _write(fig, os.path.join(outdir, "06_manhattan_outliers_by_haplotype.html"))


def plot_global_locus_methylation_ordered(
    locus_stats: pd.DataFrame,
    outdir: str,
) -> None:
    """
    Un point par locus ordonné par méthylation moyenne croissante (haplotype = 'all').
    Y = méthylation moyenne cohorte. Barres d'erreur = IQR. Couleur = chromosome.
    Permet de visualiser la distribution globale des loci et d'identifier les
    régions constamment hypo/hyperméthylées dans la cohorte.
    """
    _out = os.path.join(outdir, "07_locus_methylation_ordered.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    df = locus_stats[locus_stats["haplotype"] == "all"].copy() \
        if not locus_stats.empty else pd.DataFrame()
    if df.empty or "cohort_mean" not in df.columns:
        log.warning("[Global] Pas de données pour locus methylation ordered.")
        return

    df = (
        df.dropna(subset=["cohort_mean"])
        .sort_values("cohort_mean")
        .reset_index(drop=True)
    )
    df = df[df["chrom"].isin(CHROM_ORDER)].copy()
    df["rank"] = np.arange(len(df))

    chrom_idx = (
        df["chrom"].map({c: i for i, c in enumerate(CHROM_ORDER)})
        .fillna(-1).astype(int)
    )

    has_iqr   = "cohort_q25" in df.columns and "cohort_q75" in df.columns
    err_plus  = (df["cohort_q75"]  - df["cohort_mean"]).clip(lower=0).values if has_iqr else None
    err_minus = (df["cohort_mean"] - df["cohort_q25"]).clip(lower=0).values  if has_iqr else None

    hover = (
        "<b>" + df["island_name"].fillna(df["island_id"]) + "</b><br>"
        + df["chrom"].astype(str) + ":"
        + df["island_start"].astype(str) + "–" + df["island_end"].astype(str) + "<br>"
        + "Méth. moy. : " + df["cohort_mean"].round(3).astype(str)
        + ("<br>IQR : [" + df["cohort_q25"].round(3).astype(str)
           + " – " + df["cohort_q75"].round(3).astype(str) + "]" if has_iqr else "")
        + ("<br>N : " + df["cohort_n_samples"].astype(str)
           if "cohort_n_samples" in df.columns else "")
    )

    fig = go.Figure(go.Scatter(
        x=df["rank"], y=df["cohort_mean"],
        mode="markers",
        marker=dict(
            color=chrom_idx,
            colorscale="Turbo",
            cmin=0, cmax=len(CHROM_ORDER) - 1,
            size=4, opacity=0.80, line=dict(width=0),
            colorbar=dict(
                title="Chr",
                tickvals=list(range(0, len(CHROM_ORDER), 2)),
                ticktext=[CHROM_ORDER[i].replace("chr", "")
                          for i in range(0, len(CHROM_ORDER), 2)],
                thickness=12, len=0.65,
            ),
            showscale=True,
        ),
        error_y=dict(
            type="data", symmetric=False,
            array=err_plus.tolist(),
            arrayminus=err_minus.tolist(),
            color="rgba(150,150,150,0.25)",
            thickness=1, width=0,
        ) if has_iqr else None,
        text=hover, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        title=(
            f"Méthylation moyenne par locus – {len(df)} loci ordonnés (haplotype = all)<br>"
            "<sub>Barres d'erreur = IQR  ·  couleur = chromosome</sub>"
        ),
        xaxis=dict(title="Loci ordonnés par méthylation croissante",
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Méthylation moyenne cohorte",
                   range=[-0.02, 1.02],
                   gridcolor="rgba(200,200,200,0.4)"),
        height=520, width=1400,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        margin=dict(l=70, r=100, t=80, b=55),
        showlegend=False,
    )
    _write(fig, os.path.join(outdir, "07_locus_methylation_ordered.html"))


def plot_global_haplotypic_delta_manhattan(
    stats: pd.DataFrame,
    offsets: dict[str, int],
    outdir: str,
) -> None:
    """
    Manhattan du delta haplotypique moyen par locus.
    Pour chaque (sample, locus) : Δ = |frac_meth_hap1 − frac_meth_hap2|.
    Agrégation par locus → mean_delta sur la cohorte.
    Δ élevé = méthylation asymétrique entre les allèles (ASM / expression monoallélique).
    """
    _out = os.path.join(outdir, "08_manhattan_haplotypic_delta.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    hap_data = stats[stats["haplotype"].isin(["hap1", "hap2"])].copy()
    if hap_data.empty:
        log.warning("[Global] Pas de données hap1/hap2 – haplotypic delta manhattan ignoré.")
        return

    pivot = hap_data.pivot_table(
        index=["sample", "island_id"],
        columns="haplotype",
        values="frac_meth",
    ).reset_index()
    pivot.columns.name = None
    if "hap1" not in pivot.columns or "hap2" not in pivot.columns:
        log.warning("[Global] Colonnes hap1/hap2 manquantes – delta manhattan ignoré.")
        return

    pivot = pivot.dropna(subset=["hap1", "hap2"])
    if pivot.empty:
        return

    pivot["delta"] = (pivot["hap1"] - pivot["hap2"]).abs()

    agg = pivot.groupby("island_id").agg(
        mean_delta  =("delta", "mean"),
        median_delta=("delta", "median"),
        std_delta   =("delta", "std"),
        n_samples   =("sample", "nunique"),
    ).reset_index()

    pos_df = (
        stats[["island_id", "island_name", "chrom", "island_start", "island_end"]]
        .drop_duplicates("island_id")
    )
    df = agg.merge(pos_df, on="island_id", how="left").dropna(subset=["chrom"])
    df = df[df["chrom"].isin(CHROM_ORDER)].copy()
    df["_ci"] = df["chrom"].map({c: i for i, c in enumerate(CHROM_ORDER)})
    df = df.sort_values(["_ci", "island_start"])
    mid = (df["island_start"] + df["island_end"]) // 2
    df["x_pos"] = df["chrom"].map(offsets) + mid

    _, ticks_x, ticks_t = _manhattan_layout_elements(offsets)
    total_len = sum(CHROM_SIZES_GRCH38.values())

    fig = go.Figure()
    for ci, chrom in enumerate(CHROM_ORDER):
        pts = df[df["chrom"] == chrom]
        if pts.empty:
            continue
        color = "rgba(44,130,201,0.80)" if ci % 2 == 0 else "rgba(44,130,201,0.38)"
        size  = np.clip(3 + pts["mean_delta"] * 14, 3, 14)
        hover = (
            "<b>" + pts["island_name"].fillna(pts["island_id"]) + "</b><br>"
            + pts["island_id"] + "<br>"
            + "Δ moyen     : " + pts["mean_delta"].round(3).astype(str) + "<br>"
            + "Δ médiane   : " + pts["median_delta"].round(3).astype(str) + "<br>"
            + "Δ écart-type : " + pts["std_delta"].round(3).astype(str) + "<br>"
            + "N samples   : " + pts["n_samples"].astype(str)
        )
        fig.add_trace(go.Scatter(
            x=pts["x_pos"], y=pts["mean_delta"],
            mode="markers",
            marker=dict(color=color, size=size, opacity=1.0, line=dict(width=0)),
            text=hover, hoverinfo="text", showlegend=False,
        ))

    # Seuil indicatif Δ = 0.5 (ASM fort)
    fig.add_shape(
        type="line", xref="x", yref="y",
        x0=0, x1=total_len, y0=0.5, y1=0.5,
        line=dict(color="rgba(200,50,50,0.55)", width=1.3, dash="dash"),
    )
    fig.add_annotation(
        x=total_len * 0.995, y=0.52, text="Δ = 0.5",
        showarrow=False, font=dict(size=9, color="rgba(180,40,40,0.8)"),
        xanchor="right",
    )

    n_high = int((df["mean_delta"] >= 0.5).sum())
    log.info("  Delta haplotypique calculé sur %d loci, %d avec Δ≥0.5.", len(df), n_high)
    fig.update_layout(
        title=(
            f"Delta haplotypique moyen par locus  ·  {len(df)} loci  ·  "
            f"{n_high} avec Δ ≥ 0.5 (ASM fort)<br>"
            "<sub>Δ = |Hap1 − Hap2|, moyenné sur la cohorte  ·  "
            "Δ élevé → méthylation asymétrique entre les deux allèles</sub>"
        ),
        xaxis=dict(
            title="Position chromosomique (GRCh38)",
            tickvals=ticks_x, ticktext=ticks_t,
            tickfont=dict(size=9), showgrid=False, zeroline=False,
            range=[-total_len * 0.005, total_len * 1.01],
        ),
        yaxis=dict(
            title="Δ haplotypique moyen  |Hap1 − Hap2|",
            range=[-0.02, 1.02],
            gridcolor="rgba(200,200,200,0.4)", zeroline=False,
        ),
        height=480, width=1700,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        margin=dict(l=70, r=60, t=95, b=55),
        showlegend=False,
    )
    _write(fig, os.path.join(outdir, "08_manhattan_haplotypic_delta.html"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS PAR ÉCHANTILLON
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_volcano(
    df_s: pd.DataFrame,
    sample: str,
    outdir: str,
) -> None:
    """
    Volcano : Δ méthylation (X) vs -log10(p_value brute) (Y).
    4 subplots par haplotype. Top 10 loci les plus significatifs annotés.
    Catégories : hyper (Δ>0.1, p<0.05) | hypo (Δ<-0.1, p<0.05) |
                 sig (|Δ|≤0.1, p<0.05) | ns.
    """
    _out = os.path.join(outdir, "01_volcano.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    # Volcano : p_value brute (non ajustée)
    pcol = "p_value" if "p_value" in df_s.columns else _pval_col(df_s)
    needed = [pcol, "cohort_mean", "frac_meth", "z_score"]
    df = df_s.dropna(subset=[c for c in needed if c in df_s.columns]).copy()
    if df.empty:
        return

    df["delta"] = df["frac_meth"] - df["cohort_mean"]
    # Cap à 50 pour éviter les points parasites liés aux p-values arrondies à 0.
    # La valeur minimale utilisable est déterminée par la précision de erfc (≈1e-16).
    _MAX_LOGP = 50.0
    df["neg_logp"] = (-np.log10(df[pcol].clip(lower=1e-300))).clip(upper=_MAX_LOGP)
    df["cat"] = "ns"
    df.loc[(df[pcol] < 0.05) & (df["delta"] >  0.1), "cat"] = "hyper"
    df.loc[(df[pcol] < 0.05) & (df["delta"] < -0.1), "cat"] = "hypo"
    df.loc[(df[pcol] < 0.05) & (df["delta"].abs() <= 0.1), "cat"] = "sig (Δ<0.1)"

    CAT = {
        "hyper":    ("#cc2222", 7, 0.80),
        "hypo":     ("#2244bb", 7, 0.80),
        "sig (Δ<0.1)": ("#ff7f0e", 6, 0.75),
        "ns":       ("lightgrey", 4, 0.30),
    }

    haps_present = [h for h in HAP_ORDER if h in df["haplotype"].unique()]
    ncols = 2
    nrows = -(-len(haps_present) // ncols)
    thresh_y = -np.log10(0.05)

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[HAP_TITLES.get(h, h) for h in haps_present],
        horizontal_spacing=0.08, vertical_spacing=0.14,
    )

    for idx, hap in enumerate(haps_present):
        row = idx // ncols + 1
        col = idx % ncols + 1
        sub = df[df["haplotype"] == hap]
        top10 = sub.nsmallest(10, pcol)

        for cat, (color, size, alpha) in CAT.items():
            pts = sub[sub["cat"] == cat]
            if pts.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pts["delta"], y=pts["neg_logp"],
                mode="markers",
                marker=dict(color=color, size=size, opacity=alpha,
                            line=dict(width=0)),
                text=(
                    "<b>" + pts["island_name"].fillna(pts["island_id"]) + "</b><br>"
                    + "Δ: " + pts["delta"].round(3).astype(str) + "<br>"
                    + pcol + ": " + pts[pcol].apply(lambda v: f"{v:.2e}") + "<br>"
                    + "Z: " + pts["z_score"].round(2).astype(str) + "<br>"
                    + "Meth sample  : " + pts["frac_meth"].round(3).astype(str) + "<br>"
                    + "Meth cohorte : " + pts["cohort_mean"].round(3).astype(str)
                ),
                hoverinfo="text",
                name=cat,
                showlegend=(idx == 0),
            ), row=row, col=col)

        # Labels top 10
        if not top10.empty:
            fig.add_trace(go.Scatter(
                x=top10["delta"], y=top10["neg_logp"],
                mode="text",
                text=top10["island_name"].fillna("").astype(str),
                textposition="top center",
                textfont=dict(size=7, color="#333"),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=col)

        fig.add_hline(y=thresh_y, line_dash="dash", line_color="black",
                      line_width=1, row=row, col=col)
        for xv in (0.1, -0.1):
            fig.add_vline(x=xv, line_dash="dot", line_color="grey",
                          line_width=1, row=row, col=col)

    n_capped = int((df["neg_logp"] >= _MAX_LOGP).sum())
    cap_note = f"  ·  {n_capped} point(s) cappé(s) à {_MAX_LOGP}" if n_capped else ""

    fig.update_layout(
        title=f"Volcano – {sample}  ·  Δ méthylation vs −log₁₀({pcol}){cap_note}",
        height=440 * nrows, width=1100,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        margin=dict(l=60, r=80, t=80, b=50),
        legend=dict(title="Catégorie", orientation="v", x=1.08, y=1),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.layout[ax].update(title="Δ méthylation (sample − cohorte)",
                                  showgrid=False, zeroline=True,
                                  zerolinecolor="rgba(0,0,0,0.2)")
        if ax.startswith("yaxis"):
            fig.layout[ax].update(title=f"−log₁₀({pcol})  [cap {_MAX_LOGP}]",
                                  gridcolor="rgba(200,200,200,0.4)")
    _write(fig, os.path.join(outdir, "01_volcano.html"))


def plot_sample_hap1_vs_hap2(
    df_s: pd.DataFrame,
    sample: str,
    outdir: str,
) -> None:
    """Scatter hap1 vs hap2 méthylation pour tous les loci d'un échantillon."""
    _out = os.path.join(outdir, "02_hap1_vs_hap2.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    pivot = df_s[df_s["haplotype"].isin(["hap1", "hap2"])].pivot_table(
        index=["island_id", "island_name"], columns="haplotype", values="frac_meth"
    ).reset_index()
    pivot.columns.name = None
    if "hap1" not in pivot.columns or "hap2" not in pivot.columns:
        log.warning("  [%s] Données hap1/hap2 absentes – plot ignoré.", sample)
        return
    pivot = pivot.dropna(subset=["hap1", "hap2"])
    if pivot.empty:
        return

    pivot["delta"] = pivot["hap1"] - pivot["hap2"]
    fig = px.scatter(
        pivot, x="hap1", y="hap2", color="delta",
        color_continuous_scale="RdBu_r", range_color=[-1, 1],
        hover_data=["island_id", "island_name"],
        labels={"hap1": "Méthylation Hap1", "hap2": "Méthylation Hap2",
                "delta": "Δ (hap1−hap2)"},
        title=f"Hap1 vs Hap2 – {sample}  ({len(pivot)} loci)",
        opacity=0.70,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="grey", dash="dash"))
    fig.update_layout(
        xaxis_range=[-0.02, 1.02], yaxis_range=[-0.02, 1.02],
        height=570, width=650,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
    )
    _write(fig, os.path.join(outdir, "02_hap1_vs_hap2.html"))


def plot_sample_manhattan(
    df_s: pd.DataFrame,
    sample: str,
    offsets: dict[str, int],
    z_thresh: float,
    top_labels: int,
    haplotype: str,
    outdir: str,
) -> None:
    """
    Manhattan z_score pour un échantillon.
    Y = z_score  |  Couleur = signed −log₁₀(p_value).
    """
    _out = os.path.join(outdir, "03_manhattan.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    df = df_s[df_s["haplotype"] == haplotype].copy()
    df = df[df["chrom"].isin(CHROM_ORDER)].dropna(subset=["z_score"])
    if df.empty:
        log.warning("  [%s] Aucune donnée haplotype='%s' – manhattan ignoré.", sample, haplotype)
        return

    df["_ci"] = df["chrom"].map({c: i for i, c in enumerate(CHROM_ORDER)})
    df = df.sort_values(["_ci", "island_start"]).reset_index(drop=True)

    mid = (df["island_start"] + df["island_end"]) // 2
    df["x_pos"] = df["chrom"].map(offsets) + mid

    # Couleur = signed -log10(p)
    pcol = _pval_col(df)
    p_safe = df[pcol].clip(lower=1e-300) if pcol in df.columns else pd.Series(1.0, index=df.index)
    logp = -np.log10(p_safe)
    df["signed_logp"] = np.sign(df["z_score"]) * logp
    cmax = float(max(df["signed_logp"].abs().quantile(0.99), -np.log10(0.05)))

    is_out = df["z_score"].abs() >= z_thresh
    size   = np.where(is_out, np.clip(6 + (df["z_score"].abs() - z_thresh) * 1.5, 6, 14), 4.5)
    alpha  = np.where(is_out, 0.90, 0.50)

    shapes, ticks_x, ticks_t = _manhattan_layout_elements(offsets)
    total_len = sum(CHROM_SIZES_GRCH38.values())

    delta_str = (df["frac_meth"] - df["cohort_mean"]).apply(lambda v: f"{v:+.3f}") \
                if "cohort_mean" in df.columns else pd.Series("?", index=df.index)
    hover = (
        "<b>" + df["island_name"].fillna(df["island_id"]) + "</b><br>"
        + df["island_id"] + "<br>"
        + "Z: " + df["z_score"].round(2).astype(str) + "<br>"
        + "p: " + p_safe.apply(lambda v: f"{v:.2e}") + "<br>"
        + "Meth: " + df["frac_meth"].round(3).astype(str) + "  (Δ=" + delta_str + ")<br>"
        + "Cohorte: " + df["cohort_mean"].round(3).astype(str)
    )

    fig = go.Figure(go.Scatter(
        x=df["x_pos"], y=df["z_score"],
        mode="markers",
        marker=dict(
            color=df["signed_logp"], colorscale="RdBu_r",
            cmin=-cmax, cmax=cmax,
            size=size, opacity=alpha, line=dict(width=0),
            colorbar=dict(title="Signed<br>−log₁₀(p)", len=0.6, thickness=14, x=1.01),
            showscale=True,
        ),
        text=hover, hoverinfo="text",
    ))

    for sign, color in [(+1, "rgba(190,40,40,0.65)"), (-1, "rgba(30,80,200,0.65)")]:
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=0, x1=total_len,
            y0=sign * z_thresh, y1=sign * z_thresh,
            line=dict(color=color, width=1.3, dash="dash"),
        )

    # Annotations outliers
    if top_labels > 0 and is_out.any():
        top = (
            df[is_out]
            .assign(_abs_z=df.loc[is_out, "z_score"].abs())
            .sort_values("_abs_z", ascending=False)
            .groupby("chrom").head(top_labels)
        )
        for _, r in top.iterrows():
            label = str(r.get("island_name", r["island_id"]))
            if not label or label in ("nan", "None", ""):
                label = str(r["island_id"])
            if len(label) > 18:
                label = label[:17] + "…"
            fig.add_annotation(
                x=r["x_pos"], y=r["z_score"], text=label,
                showarrow=False,
                font=dict(size=8, color="rgba(40,40,40,0.85)"),
                yanchor="bottom" if r["z_score"] > 0 else "top",
                yshift=6 if r["z_score"] > 0 else -6,
                xanchor="center",
            )

    n_out = int(is_out.sum())
    fig.update_layout(
        title=(
            f"Manhattan – {sample}  ·  {len(df)} loci  ·  {n_out} outlier(s) "
            f"(|Z|≥{z_thresh})  ·  haplotype = {haplotype}"
        ),
        xaxis=dict(
            title="Position chromosomique (GRCh38)",
            tickvals=ticks_x, ticktext=ticks_t,
            tickfont=dict(size=9), showgrid=False, zeroline=False,
            range=[-total_len * 0.005, total_len * 1.01],
        ),
        yaxis=dict(
            title="Z-score  (↑ hyperméthylé  ·  ↓ hypométhylé)",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.20)", zerolinewidth=1.5,
            gridcolor="rgba(200,200,200,0.35)",
        ),
        shapes=shapes,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        height=520, width=1700,
        margin=dict(l=70, r=90, t=70, b=55),
        showlegend=False,
    )
    _write(fig, os.path.join(outdir, "03_manhattan.html"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS PAR LOCUS
# ─────────────────────────────────────────────────────────────────────────────

def plot_locus_boxplot(
    df_l: pd.DataFrame,
    island_id: str,
    island_name: str,
    outdir: str,
) -> None:
    """
    4 subplots (un par haplotype) : boxplot + strip des frac_meth.
    Points colorés par z_score (rouge=hyper, bleu=hypo).
    Outliers mis en évidence par une bordure noire.
    """
    _out = os.path.join(outdir, "01_boxplot_haplotypes.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    haps_present = [h for h in HAP_ORDER if h in df_l["haplotype"].unique()]
    if not haps_present:
        return

    ncols = len(haps_present)
    nrows = 1
    rng = np.random.default_rng(42)  # jitter reproductible

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[HAP_TITLES.get(h, h) for h in haps_present],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    for idx, hap in enumerate(haps_present):
        row = 1
        col = idx + 1
        sub = df_l[df_l["haplotype"] == hap].copy()
        if sub.empty:
            continue

        # Boxplot de fond (sans points intégrés)
        fig.add_trace(go.Box(
            x=[0] * len(sub),
            y=sub["frac_meth"],
            boxpoints=False,
            fillcolor=_hex_to_rgba(HAP_COLORS[hap], 0.20),
            line_color=HAP_COLORS[hap],
            showlegend=False, name=hap,
            hoverinfo="skip",
        ), row=row, col=col)

        # Tous les points individuels – couleur haplotype, jitter, hover complet
        sub_r = sub.reset_index(drop=True)
        jitter = rng.uniform(0.30, 0.70, len(sub_r))
        is_out = sub_r.get("outlier", pd.Series(False, index=sub_r.index)).fillna(False).astype(bool)
        has_z = bool("z_score" in sub_r.columns and sub_r["z_score"].notna().any())

        # Construction du hover – flag outlier + p-value dans le texte
        pcol_box = "p_value" if "p_value" in sub_r.columns else \
                   ("p_adj" if "p_adj" in sub_r.columns else None)

        def _outlier_line(row_s):
            if not row_s.get("outlier", False):
                return ""
            pval = ""
            if pcol_box and pd.notna(row_s.get(pcol_box)):
                pval = f"  –  {pcol_box} = {row_s[pcol_box]:.2e}"
            return f"<br>⚑ <b>OUTLIER</b>{pval}"

        outlier_flag = sub_r.apply(_outlier_line, axis=1)

        hover_txt = (
            "<b>" + sub_r["sample"] + "</b><br>"
            + "Meth sample  : " + sub_r["frac_meth"].round(3).astype(str) + "<br>"
            + ("Meth cohorte : " + sub_r["cohort_mean"].round(3).astype(str) + "<br>"
               if "cohort_mean" in sub_r.columns else "")
            + ("Méd. cohorte : " + sub_r["cohort_median"].round(3).astype(str) + "<br>"
               if "cohort_median" in sub_r.columns else "")
            + ("Z: " + sub_r["z_score"].round(2).astype(str) if has_z else "")
            + outlier_flag
        )

        # Tous les points – même style visuel, information dans le hover
        fig.add_trace(go.Scatter(
            x=jitter,
            y=sub_r["frac_meth"],
            mode="markers",
            marker=dict(
                color=HAP_COLORS[hap],
                size=7, opacity=1.0,
                line=dict(width=0),
            ),
            text=hover_txt,
            hoverinfo="text", showlegend=False,
        ), row=row, col=col)

        fig.update_xaxes(showticklabels=False, showgrid=False,
                         range=[-0.35, 1.05], row=row, col=col)
        fig.update_yaxes(range=[-0.02, 1.02], row=row, col=col,
                         title_text="Fraction méthylation" if col == 1 else "",
                         gridcolor="rgba(200,200,200,0.4)")

    title_str = f"{island_name} | {island_id}" if island_name and island_name not in ("nan", "") \
                else island_id
    fig.update_layout(
        title=title_str + "<br><sub>Distribution méthylation par haplotype</sub>",
        height=500, width=300 * ncols + 150,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        showlegend=False,
        margin=dict(l=65, r=110, t=90, b=40),
    )
    _write(fig, os.path.join(outdir, "01_boxplot_haplotypes.html"))


def plot_locus_hap1_vs_hap2(
    df_l: pd.DataFrame,
    island_id: str,
    island_name: str,
    outdir: str,
) -> None:
    """Scatter hap1 vs hap2 méthylation, un point par échantillon."""
    _out = os.path.join(outdir, "02_hap1_vs_hap2.html")
    if os.path.exists(_out):
        log.info("    [SKIP] fichier déjà existant : %s", _out)
        return
    pivot = df_l[df_l["haplotype"].isin(["hap1", "hap2"])].pivot_table(
        index="sample", columns="haplotype", values="frac_meth"
    ).reset_index()
    pivot.columns.name = None
    if "hap1" not in pivot.columns or "hap2" not in pivot.columns:
        return
    pivot = pivot.dropna(subset=["hap1", "hap2"])
    if pivot.empty:
        return

    pivot["delta"] = pivot["hap1"] - pivot["hap2"]
    title_str = f"{island_name} | {island_id}" if island_name and island_name not in ("nan", "") \
                else island_id
    fig = go.Figure(go.Scatter(
        x=pivot["hap1"],
        y=pivot["hap2"],
        mode="markers",
        marker=dict(
            color=pivot["delta"],
            colorscale="RdBu_r",
            cmin=-1, cmax=1,
            size=10, opacity=0.85,
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            colorbar=dict(title="Δ<br>(hap1−hap2)"),
            showscale=True,
        ),
        text=(
            "<b>" + pivot["sample"] + "</b><br>"
            + "Hap1 : " + pivot["hap1"].round(3).astype(str) + "<br>"
            + "Hap2 : " + pivot["hap2"].round(3).astype(str) + "<br>"
            + "Δ    : " + pivot["delta"].round(3).astype(str)
        ),
        hoverinfo="text",
    ))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="grey", dash="dash"))
    fig.update_layout(
        title=f"Hap1 vs Hap2 – {title_str}",
        xaxis=dict(title="Méthylation Hap1", range=[-0.02, 1.02]),
        yaxis=dict(title="Méthylation Hap2", range=[-0.02, 1.02]),
        height=530, width=640,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
    )
    _write(fig, os.path.join(outdir, "02_hap1_vs_hap2.html"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stats", required=True,
                        help="all_samples_stats.tsv (sortie de 2_cohort_analysis.py)")
    parser.add_argument("--outdir", required=True,
                        help="Répertoire de sortie des plots")
    parser.add_argument("--locus_stats", default=None,
                        help="cohort_locus_stats.tsv (auto-détecté si absent)")
    parser.add_argument("--outliers", default=None,
                        help="outliers.tsv (auto-détecté si absent)")
    parser.add_argument("--top_loci", type=int, default=100,
                        help="Nb de loci outliers pour les plots individuels (défaut : 100)")
    parser.add_argument("--z_thresh", type=float, default=2.5,
                        help="Seuil |Z| outlier (défaut : 2.5)")
    parser.add_argument("--manhattan_hap", default="all",
                        choices=["all", "hap1", "hap2", "unphased"],
                        help="Haplotype pour le Manhattan par échantillon (défaut : all)")
    parser.add_argument("--top_labels", type=int, default=3,
                        help="Nb d'outliers annotés par chromosome dans le Manhattan (défaut : 3)")
    parser.add_argument("--no_per_sample", action="store_true",
                        help="Ne pas générer les plots par échantillon")
    parser.add_argument("--no_per_loci", action="store_true",
                        help="Ne pas générer les plots par locus")
    args = parser.parse_args()

    # Auto-détection des fichiers compagnons
    stats_dir = str(Path(args.stats).parent)
    locus_path   = args.locus_stats or os.path.join(stats_dir, "cohort_locus_stats.tsv")
    outliers_path = args.outliers   or os.path.join(stats_dir, "outliers.tsv")

    log.info("=" * 60)
    log.info("  Visualisations méthylation CpG")
    log.info("=" * 60)
    log.info("  Stats TSV    : %s", args.stats)
    log.info("  Locus stats  : %s", locus_path)
    log.info("  Outliers     : %s", outliers_path)
    log.info("  Outdir       : %s", args.outdir)
    log.info("  top_loci     : %d", args.top_loci)
    log.info("  z_thresh     : %.1f", args.z_thresh)
    log.info("  manhattan_hap: %s", args.manhattan_hap)
    log.info("=" * 60)

    stats, locus_stats, outliers = load_data(args.stats, locus_path, outliers_path)

    # Exclure chrX et chrY de l'ensemble des données (une seule fois)
    _SEX_CHROMS = {"chrX", "chrY"}
    if "chrom" in stats.columns:
        stats = stats[~stats["chrom"].isin(_SEX_CHROMS)].copy()
    if not locus_stats.empty and "chrom" in locus_stats.columns:
        locus_stats = locus_stats[~locus_stats["chrom"].isin(_SEX_CHROMS)].copy()
    if not outliers.empty and "chrom" in outliers.columns:
        outliers = outliers[~outliers["chrom"].isin(_SEX_CHROMS)].copy()
    log.info("  Chromosomes sexuels (chrX, chrY) exclus de tous les plots.")

    offsets = _cumulative_offsets()

    # ── GLOBAL ───────────────────────────────────────────────────────────────
    global_dir = os.path.join(args.outdir, "01_global")
    os.makedirs(global_dir, exist_ok=True)
    log.info("[GLOBAL] 8 plots …")
    plot_global_manhattan_outlier_count(outliers, locus_stats, offsets, global_dir)
    plot_global_outlier_scatter(stats, args.z_thresh, global_dir)
    plot_global_stats_distributions(locus_stats, global_dir)
    plot_global_heatmap_outliers(stats, global_dir, top_n=args.top_loci)
    plot_global_methylation_embedding(stats, global_dir)
    plot_global_manhattan_outliers_by_haplotype(outliers, stats, offsets, global_dir)
    plot_global_locus_methylation_ordered(locus_stats, global_dir)
    plot_global_haplotypic_delta_manhattan(stats, offsets, global_dir)

    # ── PAR ÉCHANTILLON ───────────────────────────────────────────────────────
    if not args.no_per_sample:
        samples_root = os.path.join(args.outdir, "02_samples")
        os.makedirs(samples_root, exist_ok=True)
        samples = sorted(stats["sample"].unique())
        log.info("[SAMPLES] %d échantillon(s) …", len(samples))
        for sample in samples:
            sdir = os.path.join(samples_root, _sanitize(sample))
            os.makedirs(sdir, exist_ok=True)
            log.info("  %s", sample)
            df_s = stats[stats["sample"] == sample].copy()
            plot_sample_volcano(df_s, sample, sdir)
            plot_sample_hap1_vs_hap2(df_s, sample, sdir)
            plot_sample_manhattan(df_s, sample, offsets, args.z_thresh,
                                  args.top_labels, args.manhattan_hap, sdir)

    # ── PAR LOCUS ─────────────────────────────────────────────────────────────
    if not args.no_per_loci:
        loci_root = os.path.join(args.outdir, "03_loci")
        os.makedirs(loci_root, exist_ok=True)

        # Tous les loci présents dans les données, triés par chromosome
        all_islands = (
            stats[["island_id", "island_name", "chrom"]]
            .drop_duplicates("island_id")
            .copy()
        )
        all_islands = all_islands[all_islands["chrom"].isin(CHROM_ORDER)].copy()
        all_islands["_ci"] = all_islands["chrom"].map(
            {c: i for i, c in enumerate(CHROM_ORDER)}
        )
        all_islands = all_islands.sort_values(["_ci", "island_id"]).drop(columns="_ci")

        log.info("[LOCI] %d loci au total – organisés par chromosome …", len(all_islands))
        for _, lrow in all_islands.iterrows():
            island_id   = lrow["island_id"]
            island_name = str(lrow.get("island_name", ""))
            if island_name in ("nan", "None", ""):
                island_name = ""
            chrom = str(lrow["chrom"])

            # Sous-dossier par chromosome
            ldir = os.path.join(
                loci_root, chrom,
                _sanitize(f"{island_id}_{island_name}" if island_name else island_id),
            )
            os.makedirs(ldir, exist_ok=True)

            df_l = stats[stats["island_id"] == island_id].copy()
            if df_l.empty:
                continue
            log.info("  [%s] %s  [%s]", chrom, island_id, island_name)
            plot_locus_boxplot(df_l, island_id, island_name, ldir)
            plot_locus_hap1_vs_hap2(df_l, island_id, island_name, ldir)

    log.info("=" * 60)
    log.info("Terminé. Plots dans : %s", args.outdir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
