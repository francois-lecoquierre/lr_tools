#!/usr/bin/env python3
"""
2_QC.py – Contrôle qualité des échantillons (sample QC)
=========================================================
Prend en entrée le fichier consolidé produit par 1_detect_methylation.py
(all_samples_counts.tsv) et génère un rapport QC permettant d'identifier
les échantillons aberrants à exclure avant l'analyse de cohorte.

Sorties
-------
  <outdir>/01_embedding.html       – embedding 2D PCA/UMAP des profils
  <outdir>/02_qc_metrics.html      – métriques QC par échantillon
  <outdir>/sample_qc_metrics.tsv   – métriques QC tabulées
  <outdir>/samples_exclude.txt     – liste d'exclusion à éditer manuellement

Critères de suggestion d'exclusion automatique
-----------------------------------------------
  - Distance de Mahalanobis dans l'espace PCA > --mahal_thresh (défaut 3.0)
  - OU fraction de loci couverts < --min_coverage_frac       (défaut 0.5)
  - OU méthylation moyenne hors [--meth_lo, --meth_hi]       (défauts 0.1–0.9)

Le fichier samples_exclude.txt doit être relu et édité manuellement
AVANT de lancer l'étape 3 (3_cohort_analysis.py).

Usage
-----
  python 2_QC.py \\
      --counts   results/01_counts/all_samples_counts.tsv \\
      --outdir   results/02_QC \\
      [--mahal_thresh      3.0] \\
      [--min_coverage_frac 0.5] \\
      [--meth_lo           0.1] \\
      [--meth_hi           0.9] \\
      [--force]

Dépendances : pip install pandas numpy plotly scikit-learn
              (umap-learn optionnel pour UMAP)
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
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

PLOT_BGCOLOR  = "#f7f7f7"
PAPER_BGCOLOR = "#fffdfd"
SEX_CHROMS    = {"chrX", "chrY"}


def _write(fig, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")
    log.info("    → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_counts(counts_path: str) -> pd.DataFrame:
    if not os.path.isfile(counts_path):
        sys.exit(f"[ERROR] Fichier introuvable : {counts_path}")
    df = pd.read_csv(counts_path, sep="\t")
    # Exclure chromosomes sexuels (cohérent avec les autres étapes)
    if "chrom" in df.columns:
        df = df[~df["chrom"].isin(SEX_CHROMS)].copy()
    log.info("Chargement : %d lignes, %d échantillons, %d loci",
             len(df), df["sample"].nunique(), df["island_id"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Métriques QC par échantillon
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque échantillon :
      n_loci_total      – nombre de loci présents dans le fichier
      n_loci_covered    – loci couverts (haplotype='all')
      coverage_frac     – n_loci_covered / n_loci_total
      mean_meth         – méthylation moyenne (haplotype='all')
      median_meth       – médiane méthylation
      std_meth          – écart-type méthylation
      mean_reads        – profondeur moyenne
      median_reads      – médiane profondeur
      n_hap1_loci       – loci couverts en hap1
      n_hap2_loci       – loci couverts en hap2
    """
    all_loci = df["island_id"].nunique()
    df_all   = df[df["haplotype"] == "all"].copy()

    qc = df_all.groupby("sample").agg(
        n_loci_covered = ("island_id", "nunique"),
        mean_meth      = ("frac_meth",  "mean"),
        median_meth    = ("frac_meth",  "median"),
        std_meth       = ("frac_meth",  "std"),
        mean_reads     = ("n_reads",    "mean"),
        median_reads   = ("n_reads",    "median"),
    ).reset_index()

    qc["n_loci_total"]  = all_loci
    qc["coverage_frac"] = (qc["n_loci_covered"] / all_loci).round(4)

    for hap in ("hap1", "hap2"):
        h = (
            df[df["haplotype"] == hap]
            .groupby("sample")["island_id"].nunique()
            .rename(f"n_{hap}_loci")
        )
        qc = qc.merge(h, on="sample", how="left")

    for col in ("mean_meth", "median_meth", "std_meth", "mean_reads", "median_reads"):
        if col in qc.columns:
            qc[col] = qc[col].round(4)

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Embedding PCA / UMAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_embedding(df: pd.DataFrame) -> "tuple[pd.DataFrame, np.ndarray, str]":
    """
    Construit l'embedding 2D à partir de la matrice échantillons × loci
    (haplotype = 'all', frac_meth).
    Retourne (embed_df, pca_coords_2d, method_label).
    pca_coords_2d est toujours calculé (utile pour la distance de Mahalanobis).
    """
    df_all = df[df["haplotype"] == "all"].copy()
    pivot  = df_all.pivot_table(index="sample", columns="island_id", values="frac_meth")

    if pivot.shape[0] < 3:
        log.warning("Trop peu d'échantillons (%d) pour l'embedding.", pivot.shape[0])
        return pd.DataFrame(), np.array([]).reshape(0, 2), ""

    if not _SKLEARN_OK:
        log.warning("scikit-learn non disponible – embedding ignoré.")
        return pd.DataFrame(), np.array([]).reshape(0, 2), ""

    pivot_filled = pivot.fillna(pivot.median())
    X       = pivot_filled.values
    samples = pivot_filled.index.tolist()

    X_scaled = StandardScaler().fit_transform(X)

    # PCA toujours calculé (nécessaire pour la distance de Mahalanobis)
    pca = PCA(n_components=min(X.shape[0], X.shape[1], 10), random_state=42)
    pca_all       = pca.fit_transform(X_scaled)
    pca_coords_2d = pca_all[:, :2]
    ev            = pca.explained_variance_ratio_ * 100

    coords       = None
    method_label = f"PCA  (PC1 = {ev[0]:.1f} %,  PC2 = {ev[1]:.1f} %)"

    if _UMAP_OK:
        try:
            reducer = _umap_module.UMAP(n_components=2, random_state=42)
            coords  = reducer.fit_transform(X_scaled)
            method_label = "UMAP"
            log.info("  Embedding UMAP calculé (%d échantillons × %d loci).", *X.shape)
        except Exception as exc:
            log.warning("  UMAP échoué (%s) – bascule sur PCA.", exc)

    if coords is None:
        coords = pca_coords_2d
        log.info("  Embedding PCA calculé (%d échantillons × %d loci).", *X.shape)

    embed_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "sample": samples})
    return embed_df, pca_coords_2d, method_label


def _mahalanobis(pca_coords: np.ndarray) -> np.ndarray:
    """Distance de Mahalanobis dans l'espace des 2 premières CP."""
    if pca_coords.shape[0] < 3:
        return np.zeros(pca_coords.shape[0])
    center = np.mean(pca_coords, axis=0)
    cov    = np.cov(pca_coords.T)
    if np.linalg.matrix_rank(cov) < 2:
        # Matrice singulière : distance euclidienne normalisée
        std = np.std(pca_coords, axis=0, ddof=1)
        std[std == 0] = 1.0
        return np.sqrt(np.sum(((pca_coords - center) / std) ** 2, axis=1))
    cov_inv = np.linalg.inv(cov)
    diff    = pca_coords - center
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# Détection des outliers QC
# ─────────────────────────────────────────────────────────────────────────────

def flag_qc_outliers(
    qc: pd.DataFrame,
    embed_df: pd.DataFrame,
    pca_coords: np.ndarray,
    mahal_thresh: float       = 3.0,
    min_coverage_frac: float  = 0.5,
    meth_lo: float            = 0.1,
    meth_hi: float            = 0.9,
) -> pd.DataFrame:
    qc = qc.copy()

    # Distance de Mahalanobis dans l'espace PCA
    if pca_coords.shape[0] > 0 and not embed_df.empty:
        mahal     = _mahalanobis(pca_coords)
        mahal_map = dict(zip(embed_df["sample"].tolist(), mahal))
        qc["mahal_dist"]    = qc["sample"].map(mahal_map).round(3)
        qc["flag_embedding"] = qc["mahal_dist"] > mahal_thresh
    else:
        qc["mahal_dist"]    = np.nan
        qc["flag_embedding"] = False

    qc["flag_coverage"] = qc["coverage_frac"] < min_coverage_frac
    qc["flag_meth"]     = (qc["mean_meth"] < meth_lo) | (qc["mean_meth"] > meth_hi)
    qc["qc_outlier"]    = qc["flag_embedding"] | qc["flag_coverage"] | qc["flag_meth"]

    # Raisons lisibles
    qc["qc_reasons"] = ""
    qc.loc[qc["flag_embedding"], "qc_reasons"] += "embedding_outlier "
    qc.loc[qc["flag_coverage"],  "qc_reasons"] += "low_coverage "
    qc.loc[qc["flag_meth"],      "qc_reasons"] += "atypical_methylation "
    qc["qc_reasons"] = qc["qc_reasons"].str.strip()

    n_out = int(qc["qc_outlier"].sum())
    log.info("  %d échantillon(s) suggéré(s) pour exclusion.", n_out)
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_embedding(
    embed_df: pd.DataFrame,
    qc: pd.DataFrame,
    method_label: str,
    mahal_thresh: float,
    outdir: str,
) -> None:
    """Embedding 2D coloré par méthylation, outliers mis en évidence en croix rouge."""
    qc_idx      = qc.set_index("sample")
    is_outlier  = embed_df["sample"].map(lambda s: qc_idx.loc[s, "qc_outlier"] if s in qc_idx.index else False).astype(bool)
    mean_meth   = pd.to_numeric(embed_df["sample"].map(lambda s: qc_idx.loc[s, "mean_meth"]   if s in qc_idx.index else np.nan), errors="coerce")
    coverage    = pd.to_numeric(embed_df["sample"].map(lambda s: qc_idx.loc[s, "coverage_frac"] if s in qc_idx.index else np.nan), errors="coerce")
    mahal       = pd.to_numeric(embed_df["sample"].map(lambda s: qc_idx.loc[s, "mahal_dist"]  if s in qc_idx.index else np.nan), errors="coerce")
    reasons     = embed_df["sample"].map(lambda s: qc_idx.loc[s, "qc_reasons"] if s in qc_idx.index else "")

    hover = (
        "<b>" + embed_df["sample"] + "</b><br>"
        + "Méth. moyenne  : " + mean_meth.round(3).astype(str) + "<br>"
        + "Couverture     : " + (coverage * 100).round(1).astype(str) + " %<br>"
        + "Mahal. dist.   : " + mahal.round(2).astype(str) + "<br>"
        + "Statut         : " + reasons.where(reasons != "", "OK")
    )

    is_umap = method_label.startswith("UMAP")
    ax1, ax2 = ("UMAP1", "UMAP2") if is_umap else ("PC1", "PC2")
    n_out = int(is_outlier.sum())

    fig = go.Figure()

    # Points normaux
    norm = embed_df[~is_outlier]
    if not norm.empty:
        fig.add_trace(go.Scatter(
            x=norm["x"], y=norm["y"],
            mode="markers",
            marker=dict(
                color=mean_meth[~is_outlier],
                colorscale="RdBu_r", cmin=0, cmax=1,
                size=10, opacity=0.85,
                line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                colorbar=dict(title="Méth.<br>moyenne", thickness=14, len=0.55),
                showscale=True,
            ),
            text=hover[~is_outlier],
            hoverinfo="text",
            name=f"OK ({len(norm)})",
        ))

    # Points outliers
    out = embed_df[is_outlier]
    if not out.empty:
        fig.add_trace(go.Scatter(
            x=out["x"], y=out["y"],
            mode="markers",
            marker=dict(
                color="red", size=14, opacity=0.92,
                symbol="x",
                line=dict(width=2.5, color="darkred"),
            ),
            text=hover[is_outlier],
            hoverinfo="text",
            name=f"Outlier suggéré ({n_out})",
        ))

    fig.update_layout(
        title=(
            f"Embedding méthylation – {method_label}<br>"
            f"<sub>{len(embed_df)} échantillons · {n_out} outlier(s) suggéré(s)  "
            f"(Mahal. > {mahal_thresh} ou couverture/méthylation atypique) · "
            "Croix rouges = suggestion d'exclusion à vérifier manuellement</sub>"
        ),
        xaxis=dict(title=ax1, showgrid=False, zeroline=False),
        yaxis=dict(title=ax2, showgrid=False, zeroline=False),
        height=650, width=920,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=120, t=120, b=50),
    )
    _write(fig, os.path.join(outdir, "01_embedding.html"))


def plot_qc_metrics(qc: pd.DataFrame, outdir: str) -> None:
    """3 barplots : couverture, méthylation moyenne, profondeur – outliers en rouge."""
    is_out = qc["qc_outlier"].fillna(False)
    colors = ["rgba(200,40,40,0.8)" if o else "rgba(70,130,180,0.75)" for o in is_out]

    panels = [
        ("coverage_frac", "Couverture (fraction loci couverts)"),
        ("mean_meth",     "Méthylation moyenne (tous reads)"),
        ("mean_reads",    "Profondeur moyenne (reads/locus)"),
    ]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[p[1] for p in panels],
                        horizontal_spacing=0.08)

    for col_idx, (var, title) in enumerate(panels, start=1):
        if var not in qc.columns:
            continue
        fig.add_trace(go.Bar(
            x=qc["sample"], y=qc[var],
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>" + title + " : %{y}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_idx)
        fig.update_xaxes(showticklabels=False, row=1, col=col_idx)
        fig.update_yaxes(gridcolor="rgba(200,200,200,0.4)", row=1, col=col_idx)

    fig.update_layout(
        title="Métriques QC par échantillon  (barres rouges = outlier suggéré)",
        height=430, width=1300,
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PAPER_BGCOLOR,
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=30),
    )
    _write(fig, os.path.join(outdir, "02_qc_metrics.html"))


# ─────────────────────────────────────────────────────────────────────────────
# App QC interactive (HTML standalone)
# ─────────────────────────────────────────────────────────────────────────────

def write_interactive_app(
    embed_df: pd.DataFrame,
    qc: pd.DataFrame,
    method_label: str,
    mahal_thresh: float,
    outdir: str,
    counts_path: str,
) -> None:
    """
    Génère une application HTML interactive standalone (00_qc_app.html) :
      - Embedding 2D cliquable : clic pour inclure/exclure un échantillon
      - 3 barplots QC triés par ordre croissant de la valeur, mis à jour dynamiquement
      - Compteur inclus/exclus en temps réel
      - Bouton export de la liste d'exclusion en fichier texte
    """
    import json

    # ── Préparer les données ────────────────────────────────────────────────
    qc_idx = qc.set_index("sample")

    def _get(s, col, default=None):
        return qc_idx.loc[s, col] if s in qc_idx.index else default

    records = []
    for _, row in embed_df.iterrows():
        s = row["sample"]
        records.append({
            "sample":        s,
            "x":             float(row["x"]),
            "y":             float(row["y"]),
            "mean_meth":     float(_get(s, "mean_meth",     0.0)),
            "coverage_frac": float(_get(s, "coverage_frac", 0.0)),
            "mean_reads":    float(_get(s, "mean_reads",    0.0)),
            "mahal_dist":    float(_get(s, "mahal_dist",    0.0)),
            "qc_reasons":    str(_get(s,  "qc_reasons",    "")),
            "auto_exclude":  bool(_get(s, "qc_outlier",    False)),
        })

    is_umap = method_label.startswith("UMAP")
    ax1 = "UMAP1" if is_umap else "PC1"
    ax2 = "UMAP2" if is_umap else "PC2"
    n_total = len(records)
    n_auto  = sum(1 for r in records if r["auto_exclude"])

    data_json         = json.dumps(records, ensure_ascii=False)
    method_label_json = json.dumps(method_label)
    ax1_json          = json.dumps(ax1)
    ax2_json          = json.dumps(ax2)
    counts_path_json  = json.dumps(counts_path)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>QC – Méthylation CpG</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif;}}
  body{{background:#f0f2f5;color:#222;}}
  h1{{padding:16px 24px 4px;font-size:1.25rem;font-weight:600;color:#1a1a2e;}}
  .subtitle{{padding:0 24px 12px;font-size:.85rem;color:#555;}}
  .layout{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:12px;padding:12px 20px;}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.1);overflow:hidden;}}
  .card-title{{padding:10px 16px 4px;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#444;border-bottom:1px solid #eee;}}
  #div-embed{{grid-column:1/2;grid-row:1/3;}}
  #div-stats{{grid-column:2/3;grid-row:1/2;}}
  #div-bars {{grid-column:1/3;grid-row:3;}}
  .stats-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:14px;}}
  .stat-box{{background:#f7f8fc;border-radius:8px;padding:12px 10px;text-align:center;}}
  .stat-val{{font-size:1.6rem;font-weight:700;line-height:1.1;}}
  .stat-lbl{{font-size:.72rem;color:#666;margin-top:3px;}}
  .val-ok   {{color:#2a7dd1;}}
  .val-excl {{color:#c0392b;}}
  .val-auto {{color:#e67e22;}}
  .actions{{padding:10px 14px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;}}
  button{{padding:7px 16px;border:none;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:600;transition:opacity .15s;}}
  button:hover{{opacity:.82;}}
  #btn-export {{background:#2a7dd1;color:#fff;}}
  #btn-reset  {{background:#e0e0e0;color:#333;}}
  #btn-toggle {{background:#e8f4fd;color:#2a7dd1;border:1px solid #b3d7f5;}}
  .hint{{font-size:.75rem;color:#888;padding:0 14px 10px;}}
  #status-bar{{position:fixed;bottom:20px;right:24px;z-index:9999;font-size:.82rem;padding:8px 18px;background:#fff7e0;border:1px solid #ffe08a;border-radius:8px;color:#7a5800;box-shadow:0 2px 8px rgba(0,0,0,.18);opacity:0;transition:opacity .25s;pointer-events:none;}}
  #status-bar.visible{{opacity:1;}}
</style>
</head>
<body>
<h1>Application QC – Méthylation CpG</h1>
<div class="subtitle">Source : {counts_path} &nbsp;·&nbsp; Embedding : {method_label}</div>
<div class="layout">
  <div class="card" id="div-embed">
    <div class="card-title">Embedding {method_label} — cliquer sur un point pour inclure/exclure</div>
    <div id="plot-embed" style="height:480px;"></div>
    <p class="hint">Clic gauche = basculer l'état d'exclusion &nbsp;·&nbsp; Molette = zoom &nbsp;·&nbsp; Glisser = pan</p>
  </div>
  <div class="card" id="div-stats">
    <div class="card-title">Résumé de la sélection</div>
    <div class="stats-grid">
      <div class="stat-box"><div class="stat-val" id="sv-total">{n_total}</div><div class="stat-lbl">Échantillons total</div></div>
      <div class="stat-box"><div class="stat-val val-ok" id="sv-kept">–</div><div class="stat-lbl">Inclus dans l'analyse</div></div>
      <div class="stat-box"><div class="stat-val val-excl" id="sv-excl">–</div><div class="stat-lbl">Exclus manuellement</div></div>
      <div class="stat-box"><div class="stat-val val-auto" id="sv-auto">{n_auto}</div><div class="stat-lbl">Outliers auto-détectés</div></div>
      <div class="stat-box"><div class="stat-val" id="sv-meth-mean">–</div><div class="stat-lbl">Méth. moy. (inclus)</div></div>
      <div class="stat-box"><div class="stat-val" id="sv-cov-mean">–</div><div class="stat-lbl">Couverture moy. (inclus)</div></div>
    </div>
    <div class="actions">
      <button id="btn-export">⬇ Exporter la liste d'exclusion</button>
      <button id="btn-reset">↺ Réinitialiser</button>
      <button id="btn-toggle">⊘ Exclure tous les auto-détectés</button>
    </div>
    <div id="excl-list" style="padding:0 14px 12px;max-height:160px;overflow-y:auto;font-size:.78rem;color:#c0392b;line-height:1.7;"></div>
  </div>
  <div class="card" id="div-bars">
    <div class="card-title">Métriques QC par échantillon (triées par valeur croissante — rouge = exclu)</div>
    <div id="plot-bars" style="height:320px;"></div>
  </div>
</div>

<script>
// ── Données ──────────────────────────────────────────────────────────────────
const DATA        = {data_json};
const METHOD      = {method_label_json};
const AX1         = {ax1_json};
const AX2         = {ax2_json};
const SOURCE_PATH = {counts_path_json};

// État d'exclusion courant : map sample → bool (true = exclu)
const excluded = {{}};
DATA.forEach(d => {{ excluded[d.sample] = d.auto_exclude; }});

// ── Couleurs ─────────────────────────────────────────────────────────────────
const COL_OK   = 'rgba(70,130,180,0.82)';
const COL_EXCL = 'rgba(192,57,43,0.90)';
const COL_AUTO = 'rgba(230,126,34,0.85)';  // auto-détecté mais pas encore changé

// ── Utilitaires ───────────────────────────────────────────────────────────────
function mean(arr){{
  if(!arr.length) return NaN;
  return arr.reduce((a,b)=>a+b,0)/arr.length;
}}

// ── Construction des traces d'embedding ──────────────────────────────────────
function buildEmbedTraces(){{
  // 3 groupes : OK, exclu-auto (outlier suggéré), exclu-manuel
  const grp = {{ok:[], auto:[], manual:[]}};
  DATA.forEach(d => {{
    if(!excluded[d.sample])               grp.ok.push(d);
    else if(d.auto_exclude)               grp.auto.push(d);
    else                                  grp.manual.push(d);
  }});

  function makeTrace(pts, name, color, symbol, size){{
    return {{
      type:'scatter', mode:'markers',
      x: pts.map(d=>d.x), y: pts.map(d=>d.y),
      name, customdata: pts.map(d=>d.sample),
      marker:{{color, symbol, size, opacity:0.88,
               line:{{width:symbol==='circle'?0.5:2, color:'rgba(0,0,0,0.25)'}}}},
      text: pts.map(d =>
        '<b>'+d.sample+'</b><br>'+
        'Méth. moy. : '+d.mean_meth.toFixed(3)+'<br>'+
        'Couverture : '+(d.coverage_frac*100).toFixed(1)+' %<br>'+
        'Mahal. dist. : '+d.mahal_dist.toFixed(2)+'<br>'+
        'Statut : '+(excluded[d.sample]?'EXCLU':'OK')+(d.qc_reasons?' ['+d.qc_reasons+']':'')
      ),
      hoverinfo:'text',
    }};
  }}

  return [
    makeTrace(grp.ok,     'Inclus (' +grp.ok.length+')',          COL_OK,   'circle',  10),
    makeTrace(grp.auto,   'Exclu auto ('+grp.auto.length+')',     COL_AUTO, 'x',       13),
    makeTrace(grp.manual, 'Exclu manuel ('+grp.manual.length+')', COL_EXCL, 'x',       13),
  ];
}}

const embedLayout = {{
  xaxis:{{title:AX1, showgrid:false, zeroline:false}},
  yaxis:{{title:AX2, showgrid:false, zeroline:false}},
  legend:{{orientation:'h', y:1.05, x:0}},
  plot_bgcolor:'#f7f7f7', paper_bgcolor:'#fffdfd',
  margin:{{l:50,r:20,t:30,b:50}},
  hovermode:'closest',
  dragmode:'pan',
}};

// ── Construction des barplots ─────────────────────────────────────────────────
const BAR_PANELS = [
  {{key:'coverage_frac', label:'Couverture (fraction loci couverts)'}},
  {{key:'mean_meth',     label:'Méthylation moyenne'}},
  {{key:'mean_reads',    label:'Profondeur moyenne (reads/locus)'}},
];

function buildBarTraces(){{
  const traces = [];
  BAR_PANELS.forEach((panel, pi) => {{
    // Trier les échantillons par valeur croissante pour ce panel
    const sorted = [...DATA].sort((a,b) => a[panel.key] - b[panel.key]);
    traces.push({{
      type:'bar', name:panel.label,
      x: sorted.map(d=>d.sample),
      y: sorted.map(d=>d[panel.key]),
      xaxis: pi===0?'x':'x'+(pi+1),
      yaxis: pi===0?'y':'y'+(pi+1),
      marker:{{color: sorted.map(d => excluded[d.sample] ? COL_EXCL : COL_OK)}},
      hovertemplate:'<b>%{{x}}</b><br>'+panel.label+' : %{{y:.4f}}<extra></extra>',
      showlegend:false,
    }});
  }});
  return traces;
}}

const barLayout = {{
  grid:{{rows:1, columns:3, pattern:'independent'}},
  plot_bgcolor:'#f7f7f7', paper_bgcolor:'#fffdfd',
  margin:{{l:50,r:20,t:30,b:80}},
  annotations:[
    {{text:'Couverture (fraction loci couverts)', xref:'x domain', yref:'y domain',
      x:0.5, y:1.08, showarrow:false, font:{{size:11}}, xanchor:'center'}},
    {{text:'Méthylation moyenne', xref:'x2 domain', yref:'y2 domain',
      x:0.5, y:1.08, showarrow:false, font:{{size:11}}, xanchor:'center'}},
    {{text:'Profondeur moyenne (reads/locus)', xref:'x3 domain', yref:'y3 domain',
      x:0.5, y:1.08, showarrow:false, font:{{size:11}}, xanchor:'center'}},
  ],
  xaxis: {{showticklabels:false, gridcolor:'rgba(200,200,200,0.4)'}},
  xaxis2:{{showticklabels:false, gridcolor:'rgba(200,200,200,0.4)'}},
  xaxis3:{{showticklabels:false, gridcolor:'rgba(200,200,200,0.4)'}},
  yaxis: {{gridcolor:'rgba(200,200,200,0.4)'}},
  yaxis2:{{gridcolor:'rgba(200,200,200,0.4)'}},
  yaxis3:{{gridcolor:'rgba(200,200,200,0.4)'}},
}};

const cfg = {{responsive:true, scrollZoom:true,
             displayModeBar:true, modeBarButtonsToRemove:['lasso2d','select2d']}};

// ── Rendu initial ─────────────────────────────────────────────────────────────
Plotly.newPlot('plot-embed', buildEmbedTraces(), embedLayout, cfg);
Plotly.newPlot('plot-bars',  buildBarTraces(),   barLayout,   {{...cfg, scrollZoom:false}});
updateStats();

// ── Gestion du clic sur l'embedding ──────────────────────────────────────────
document.getElementById('plot-embed').on('plotly_click', function(evtData){{
  const pt = evtData.points[0];
  if(!pt) return;
  const sample = pt.customdata;
  if(!sample) return;
  excluded[sample] = !excluded[sample];
  refresh(sample);
}});

function refresh(changedSample){{
  Plotly.react('plot-embed', buildEmbedTraces(), embedLayout, cfg);
  Plotly.react('plot-bars',  buildBarTraces(),   barLayout,   {{...cfg, scrollZoom:false}});
  updateStats();
  showStatus(changedSample);
}}

function updateStats(){{
  const excl   = DATA.filter(d => excluded[d.sample]);
  const kept   = DATA.filter(d => !excluded[d.sample]);
  document.getElementById('sv-kept').textContent  = kept.length;
  document.getElementById('sv-excl').textContent  = excl.length;
  document.getElementById('sv-meth-mean').textContent =
    kept.length ? mean(kept.map(d=>d.mean_meth)).toFixed(3) : '–';
  document.getElementById('sv-cov-mean').textContent  =
    kept.length ? (mean(kept.map(d=>d.coverage_frac))*100).toFixed(1)+'%' : '–';

  const listEl = document.getElementById('excl-list');
  const names  = excl.map(d=>d.sample).sort();
  listEl.innerHTML = names.length
    ? names.map(s=>'<span style="display:inline-block;margin:1px 6px 1px 0;padding:1px 6px;background:#fde8e6;border-radius:4px;">'+s+'</span>').join('')
    : '<span style="color:#aaa">Aucun échantillon exclu</span>';
}}

function showStatus(sample){{
  const bar = document.getElementById('status-bar');
  const state = excluded[sample] ? 'EXCLU' : 'INCLUS';
  bar.textContent = '→ ' + sample + ' : ' + state;
  bar.classList.add('visible');
  clearTimeout(bar._timer);
  bar._timer = setTimeout(()=>{{bar.classList.remove('visible');}}, 2200);
}}

// ── Bouton export ─────────────────────────────────────────────────────────────
document.getElementById('btn-export').addEventListener('click', function(){{
  const excl  = DATA.filter(d => excluded[d.sample]).map(d=>d.sample).sort();
  const lines = [
    '# ============================================================',
    '# Fichier d\\'exclusion – pipeline méthylation CpG',
    '# Généré par l\\'app QC interactive (00_qc_app.html)',
    '# Source : ' + SOURCE_PATH,
    '# ============================================================',
    '#',
    '# Échantillons exclus : ' + excl.length + ' / ' + DATA.length,
    '#',
  ].concat(excl.map(s => s + '  # exclu manuellement'));
  const blob = new Blob([lines.join('\\n')+'\\n'], {{type:'text/plain'}});
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'samples_exclude.txt';
  a.click();
  URL.revokeObjectURL(a.href);
}});

// ── Bouton réinitialiser ──────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', function(){{
  DATA.forEach(d => {{ excluded[d.sample] = d.auto_exclude; }});
  Plotly.react('plot-embed', buildEmbedTraces(), embedLayout, cfg);
  Plotly.react('plot-bars',  buildBarTraces(),   barLayout,   {{...cfg, scrollZoom:false}});
  updateStats();
  document.getElementById('status-bar').classList.remove('visible');
}});

// ── Bouton exclure tous les auto-détectés ─────────────────────────────────────
document.getElementById('btn-toggle').addEventListener('click', function(){{
  const autoExcl = DATA.filter(d=>d.auto_exclude);
  const allOn    = autoExcl.every(d=>excluded[d.sample]);
  autoExcl.forEach(d => {{ excluded[d.sample] = !allOn; }});
  this.textContent = allOn ? '⊘ Exclure tous les auto-détectés' : '✓ Réinclure les auto-détectés';
  Plotly.react('plot-embed', buildEmbedTraces(), embedLayout, cfg);
  Plotly.react('plot-bars',  buildBarTraces(),   barLayout,   {{...cfg, scrollZoom:false}});
  updateStats();
}});
</script>
<div id="status-bar"></div>
</body>
</html>"""

    app_path = os.path.join(outdir, "00_qc_app.html")
    with open(app_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("App QC interactive → %s", app_path)


# ─────────────────────────────────────────────────────────────────────────────
# Écriture des sorties
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(qc: pd.DataFrame, outdir: str, counts_path: str) -> None:
    # TSV métriques
    qc_path = os.path.join(outdir, "sample_qc_metrics.tsv")
    qc.to_csv(qc_path, sep="\t", index=False)
    log.info("Métriques QC → %s", qc_path)

    # Fichier d'exclusion
    excl_path  = os.path.join(outdir, "samples_exclude.txt")
    suggested  = qc[qc["qc_outlier"]]["sample"].tolist()
    all_samples = qc["sample"].tolist()

    lines = [
        "# ============================================================",
        "# Fichier d'exclusion – pipeline méthylation CpG",
        "# ============================================================",
        "#",
        "# INSTRUCTIONS :",
        "#   1. Ouvrir 01_embedding.html et 02_qc_metrics.html pour",
        "#      inspecter visuellement les échantillons.",
        "#   2. Éditer ce fichier : décommenter ou ajouter les",
        "#      échantillons à exclure (une ligne = un nom exact).",
        "#   3. Les lignes commençant par '#' sont ignorées.",
        "#",
        f"# Généré automatiquement par 2_QC.py",
        f"# Source : {counts_path}",
        f"# Échantillons analysés : {len(all_samples)}",
        f"# Outliers suggérés    : {len(suggested)}",
        "#",
        "# ── Suggestions automatiques (déjà décommentées) ──────────",
    ]

    if suggested:
        for s in suggested:
            row     = qc[qc["sample"] == s].iloc[0]
            reasons = row.get("qc_reasons", "")
            lines.append(f"{s}  # {reasons}")
    else:
        lines.append("# (aucun outlier détecté automatiquement)")

    lines += [
        "#",
        "# ── Autres échantillons (à décommenter pour exclure) ──────",
    ]
    for s in all_samples:
        if s not in suggested:
            lines.append(f"# {s}")
    lines.append("")

    with open(excl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log.info("Fichier d'exclusion → %s", excl_path)

    if suggested:
        log.info("  → %d échantillon(s) suggéré(s) : %s",
                 len(suggested), ", ".join(suggested))
    else:
        log.info("  → Aucun outlier détecté. Vérifier manuellement les plots.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--counts", required=True,
                        help="Fichier TSV consolidé (all_samples_counts.tsv)")
    parser.add_argument("--outdir", required=True,
                        help="Répertoire de sortie QC")
    parser.add_argument("--mahal_thresh", type=float, default=3.0,
                        help="Seuil distance de Mahalanobis pour flag outlier (défaut : 3.0)")
    parser.add_argument("--min_coverage_frac", type=float, default=0.5,
                        help="Fraction minimale de loci couverts (défaut : 0.5)")
    parser.add_argument("--meth_lo", type=float, default=0.1,
                        help="Méthylation moyenne minimale acceptée (défaut : 0.1)")
    parser.add_argument("--meth_hi", type=float, default=0.9,
                        help="Méthylation moyenne maximale acceptée (défaut : 0.9)")
    parser.add_argument("--force", action="store_true",
                        help="Recalculer même si les fichiers de sortie existent déjà")
    args = parser.parse_args()

    out_qc = os.path.join(args.outdir, "sample_qc_metrics.tsv")
    if not args.force and os.path.isfile(out_qc):
        log.info("Sortie déjà existante : %s\nUtilisez --force pour recalculer.", out_qc)
        sys.exit(0)

    log.info("=" * 60)
    log.info("  QC échantillons – méthylation CpG")
    log.info("=" * 60)
    log.info("  Counts TSV      : %s", args.counts)
    log.info("  Outdir          : %s", args.outdir)
    log.info("  mahal_thresh    : %.1f", args.mahal_thresh)
    log.info("  min_coverage    : %.2f", args.min_coverage_frac)
    log.info("  meth range      : [%.2f – %.2f]", args.meth_lo, args.meth_hi)
    log.info("=" * 60)

    os.makedirs(args.outdir, exist_ok=True)

    df  = load_counts(args.counts)
    qc  = compute_sample_qc(df)

    log.info("Calcul de l'embedding …")
    embed_df, pca_coords, method_label = compute_embedding(df)

    log.info("Détection des outliers QC …")
    qc = flag_qc_outliers(
        qc, embed_df, pca_coords,
        mahal_thresh       = args.mahal_thresh,
        min_coverage_frac  = args.min_coverage_frac,
        meth_lo            = args.meth_lo,
        meth_hi            = args.meth_hi,
    )

    log.info("Génération des plots …")
    if not embed_df.empty:
        # App interactive principale (embedding + barplots + export)
        write_interactive_app(
            embed_df, qc, method_label, args.mahal_thresh,
            args.outdir, args.counts,
        )
        # Plots statiques de rétrocompatibilité
        plot_embedding(embed_df, qc, method_label, args.mahal_thresh, args.outdir)
    plot_qc_metrics(qc, args.outdir)

    write_outputs(qc, args.outdir, args.counts)

    log.info("=" * 60)
    log.info("Terminé. Résultats dans : %s", args.outdir)
    log.info("=" * 60)
    log.info("PROCHAINE ÉTAPE :")
    log.info("  1. Ouvrir   %s/00_qc_app.html  ← app interactive", args.outdir)
    log.info("  2. Inclure/exclure les échantillons dans l'app puis exporter samples_exclude.txt")
    log.info("  3. Lancer   python 3_cohort_analysis.py --exclude_samples %s/samples_exclude.txt …",
             args.outdir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
