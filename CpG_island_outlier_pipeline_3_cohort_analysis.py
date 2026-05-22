#!/usr/bin/env python3
"""
3_cohort_analysis.py – Analyse statistique de méthylation par cohorte
=======================================================================
Prend en entrée le fichier consolidé produit par 1_detect_methylation.py
(all_samples_counts.tsv) et calcule pour chaque entrée
(sample × locus × haplotype) sa déviation par rapport au cohorte.

Les échantillons à exclure peuvent être listés dans un fichier texte
(--exclude_samples), typiquement généré par l'étape 2 (2_QC.py).

Stratégie de comparaison
-------------------------
Pour chaque groupe (island_id × haplotype), les statistiques du cohorte
sont calculées sur TOUS les échantillons ayant passé les filtres qualité.
Option --leave_one_out : les stats sont recalculées en excluant l'individu
testé (évite l'auto-influence, recommandé pour cohortes < 30 individus).

Filtres qualité
---------------
  haplotype 'all'               : n_reads >= --min_reads     (défaut 15)
  haplotypes hap1/hap2/unphased : n_reads >= --min_reads_hap (défaut 8)
  tous groupes                  : island_cpgNum >= --min_cpg (défaut 10)
  tous groupes                  : n_samples >= --min_samples  (défaut 5)
    → un locus avec < min_samples individus n'est pas analysable

Colonnes ajoutées
-----------------
  cohort_n_samples    – nombre d'échantillons valides au locus
  cohort_mean         – moyenne cohorte  frac_meth
  cohort_std          – écart-type cohorte
  cohort_median       – médiane cohorte
  cohort_q25 / q75    – quartiles
  cohort_IQR          – Q75 − Q25
  z_score             – (frac_meth − cohort_mean) / cohort_std
  p_value             – p bilatérale exacte (erfc ; scipy optionnel)
  p_adj               – p-value corrigée FDR (Benjamini-Hochberg)
  IQR_outlier         – bool : en-dehors de [Q25−1.5×IQR, Q75+1.5×IQR]
  Z_outlier           – bool : |z_score| >= --z_thresh
  outlier             – bool : IQR_outlier OR Z_outlier

Sorties
-------
  <outdir>/all_samples_stats.tsv      – fichier d'entrée enrichi (toutes lignes)
  <outdir>/cohort_locus_stats.tsv     – une ligne par (locus × haplotype)
  <outdir>/outliers.tsv               – sous-ensemble des lignes outlier=True

Usage
-----
  python 3_cohort_analysis.py \\
      --counts          results/01_counts/all_samples_counts.tsv \\
      --outdir          results/03_stats \\
      [--exclude_samples results/02_QC/samples_exclude.txt] \\
      [--min_reads      15]  \\
      [--min_reads_hap   8]  \\
      [--min_cpg        10]  \\
      [--min_samples     5]  \\
      [--z_thresh      2.5]  \\
      [--leave_one_out]      \\
      [--force]

Format du fichier d'exclusion
------------------------------
  - Une ligne par échantillon à exclure (nom exact du sample)
  - Les lignes vides et les lignes commençant par '#' sont ignorées
  - Exemple :
      # Commentaire
      EFA_0005_MOR_001
      EFA_0042_BRU_001

Dépendances
-----------
  pip install pandas numpy scipy
  (scipy optionnel mais recommandé pour les p-valeurs)
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

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
# Lecture du fichier d'exclusion
# ─────────────────────────────────────────────────────────────────────────────

def read_exclude_list(path: str) -> list[str]:
    """
    Lit le fichier d'exclusion et retourne la liste des samples à exclure.
    Les lignes vides et commençant par '#' sont ignorées.
    """
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Fichier d'exclusion introuvable : {path}")
    excluded = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            # Supprimer les commentaires en fin de ligne (# ...)
            line = line.split("#")[0].strip()
            if line:
                excluded.append(line)
    log.info("Fichier d'exclusion : %d échantillon(s) à exclure → %s",
             len(excluded), path)
    if excluded:
        log.info("  %s", ", ".join(excluded))
    return excluded


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement et filtrage qualité
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter(
    counts_path: str,
    min_reads: int     = 15,
    min_reads_hap: int = 8,
    min_cpg: int       = 10,
    exclude_samples: list[str] | None = None,
) -> pd.DataFrame:
    """
    Charge le TSV consolidé, applique les filtres de profondeur,
    et exclut les échantillons listés dans exclude_samples.
    """
    log.info("Chargement de %s …", counts_path)
    df = pd.read_csv(counts_path, sep="\t")

    required = {
        "sample", "island_id", "island_name", "chrom",
        "island_start", "island_end", "island_cpgNum",
        "haplotype", "n_reads", "n_cpg_total",
        "n_cpg_meth", "n_cpg_unmeth", "frac_meth",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Colonnes manquantes dans {counts_path} : {missing}")

    log.info("  Entrées brutes : %d  (%d échantillons, %d loci, %d groupes haplotype)",
             len(df), df["sample"].nunique(),
             df["island_id"].nunique(), df["haplotype"].nunique())

    # Exclusion des échantillons QC
    if exclude_samples:
        # Vérification que les samples existent bien dans les données
        found     = [s for s in exclude_samples if s in df["sample"].unique()]
        not_found = [s for s in exclude_samples if s not in df["sample"].unique()]
        if not_found:
            log.warning("  Échantillons non trouvés dans les données (ignorés) : %s",
                        ", ".join(not_found))
        if found:
            df = df[~df["sample"].isin(found)].copy()
            log.info("  Après exclusion QC (%d échantillons) : %d échantillons restants.",
                     len(found), df["sample"].nunique())

    # Filtres de profondeur et CpG
    depth_ok = (
        ((df["haplotype"] == "all")  & (df["n_reads"] >= min_reads)) |
        ((df["haplotype"] != "all")  & (df["n_reads"] >= min_reads_hap))
    )
    filt = df[depth_ok & (df["island_cpgNum"] >= min_cpg)].copy()

    n_removed = len(df) - len(filt)
    log.info("  Après filtrage (min_reads=%d, min_reads_hap=%d, min_cpg=%d) : "
             "%d entrées conservées, %d supprimées.",
             min_reads, min_reads_hap, min_cpg, len(filt), n_removed)
    return filt


# ─────────────────────────────────────────────────────────────────────────────
# 2. Statistiques cohorte par (locus × haplotype)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cohort_stats(df: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    """
    Pour chaque (island_id × haplotype), calcule les statistiques cohorte.
    Les groupes avec < min_samples individus sont exclus (NaN dans les stats).
    """
    log.info("Calcul des statistiques cohorte (min_samples=%d) …", min_samples)

    stats = (
        df.groupby(["island_id", "island_name", "chrom",
                    "island_start", "island_end", "island_cpgNum", "haplotype"])
        .agg(
            cohort_n_samples = ("sample",    "nunique"),
            cohort_mean      = ("frac_meth", "mean"),
            cohort_std       = ("frac_meth", "std"),
            cohort_median    = ("frac_meth", "median"),
            cohort_q25       = ("frac_meth", lambda x: x.quantile(0.25)),
            cohort_q75       = ("frac_meth", lambda x: x.quantile(0.75)),
            cohort_min       = ("frac_meth", "min"),
            cohort_max       = ("frac_meth", "max"),
        )
        .reset_index()
    )
    stats["cohort_IQR"] = stats["cohort_q75"] - stats["cohort_q25"]

    # Masquer les loci avec trop peu de données
    too_few = stats["cohort_n_samples"] < min_samples
    stat_cols = ["cohort_mean", "cohort_std", "cohort_median",
                 "cohort_q25", "cohort_q75", "cohort_IQR"]
    stats.loc[too_few, stat_cols] = np.nan

    n_excluded = int(too_few.sum())
    log.info("  %d locus × haplotype avec stats valides, %d exclus (< %d échantillons).",
             len(stats) - n_excluded, n_excluded, min_samples)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Calcul du Z-score (standard ou leave-one-out)
# ─────────────────────────────────────────────────────────────────────────────

def _erfc_pvalue(z_abs: np.ndarray) -> np.ndarray:
    """p-valeur bilatérale exacte depuis |z| via erfc."""
    try:
        from scipy.special import erfc as _erfc
        return np.minimum(1.0, _erfc(z_abs / np.sqrt(2)))
    except ImportError:
        import math as _math
        log.debug("scipy absent — math.erfc utilisé.")
        return np.minimum(1.0, np.vectorize(_math.erfc)(z_abs / np.sqrt(2)))


def score_standard(
    df: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Z-score calculé avec les statistiques globales du cohorte
    (l'individu testé est inclus dans la moyenne de référence).
    """
    scored = df.merge(
        stats[["island_id", "haplotype",
               "cohort_n_samples", "cohort_mean", "cohort_std",
               "cohort_median", "cohort_q25", "cohort_q75", "cohort_IQR"]],
        on=["island_id", "haplotype"],
        how="left",
    )
    std_safe = scored["cohort_std"].replace(0, np.nan)
    scored["z_score"] = (
        (scored["frac_meth"] - scored["cohort_mean"]) / std_safe
    ).round(4)
    return scored


def score_leave_one_out(
    df: pd.DataFrame,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Z-score leave-one-out : pour chaque (sample, island_id, haplotype),
    les statistiques de référence sont calculées sur les AUTRES individus.
    """
    log.info("  Mode leave-one-out : recalcul des stats par exclusion de chaque individu …")

    key = ["island_id", "haplotype"]

    grp = df.groupby(key).agg(
        _n    = ("frac_meth", "count"),
        _mean = ("frac_meth", "mean"),
        _var  = ("frac_meth", "var"),    # diviseur n-1
    ).reset_index()

    scored = df.merge(grp, on=key, how="left")

    n       = scored["_n"]
    mu      = scored["_mean"]
    var_all = scored["_var"].fillna(0)
    x       = scored["frac_meth"]

    valid  = n >= min_samples
    mu_loo = np.where(valid & (n > 1), (n * mu - x) / (n - 1), np.nan)

    sum_sq      = (n - 1) * var_all + n * mu ** 2
    var_loo_num = sum_sq - n * mu ** 2 - (x - mu_loo) ** 2
    var_loo = np.where(
        valid & (n > 2),
        np.maximum(0, var_loo_num / (n - 2)),
        np.nan,
    )
    std_loo = np.sqrt(var_loo)

    scored["cohort_n_samples"] = n
    scored["cohort_mean"]      = mu_loo.round(6)
    scored["cohort_std"]       = std_loo.round(6)

    global_stats = df.groupby(key).agg(
        cohort_median = ("frac_meth", "median"),
        cohort_q25    = ("frac_meth", lambda x: x.quantile(0.25)),
        cohort_q75    = ("frac_meth", lambda x: x.quantile(0.75)),
    ).reset_index()
    global_stats["cohort_IQR"] = global_stats["cohort_q75"] - global_stats["cohort_q25"]

    scored = scored.merge(global_stats, on=key, how="left")

    std_safe = pd.Series(std_loo).replace(0, np.nan)
    scored["z_score"] = (
        (scored["frac_meth"] - scored["cohort_mean"]) / std_safe.values
    ).round(4)

    scored.drop(columns=["_n", "_mean", "_var"], inplace=True)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# 4. P-value, FDR, flags outliers
# ─────────────────────────────────────────────────────────────────────────────

def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Correction Benjamini-Hochberg vectorisée. Les NaN sont propagés."""
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()

    valid  = ~np.isnan(pvalues)
    padj   = np.full(n, np.nan)
    pv     = pvalues[valid]
    m      = len(pv)
    if m == 0:
        return padj

    order        = np.argsort(pv)
    ranks        = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    padj_v       = np.minimum(1.0, pv * m / ranks)
    padj_v       = np.minimum.accumulate(padj_v[::-1])[::-1]

    padj[valid] = padj_v
    return padj


def add_pvalue_and_flags(
    scored: pd.DataFrame,
    z_thresh: float = 2.5,
) -> pd.DataFrame:
    """Ajoute p_value, p_adj (BH par groupe haplotype), IQR_outlier, Z_outlier, outlier."""
    log.info("Calcul des p-valeurs et correction FDR (BH par groupe haplotype) …")

    scored = scored.copy()

    z_abs = scored["z_score"].abs().values
    scored["p_value"] = _erfc_pvalue(np.where(np.isnan(z_abs), np.nan, z_abs))

    scored["p_adj"] = np.nan
    for hap, idx in scored.groupby("haplotype").groups.items():
        pv = scored.loc[idx, "p_value"].values.astype(float)
        scored.loc[idx, "p_adj"] = _bh_fdr(pv)

    # Ne pas arrondir les p-values : round(6) transforme les très petites
    # p-values (ex. 2e-9) en 0.0, ce qui génère -log10(0) = ∞ dans les volcano plots.
    # Les valeurs sont conservées en précision float64 complète dans le TSV.

    iqr = scored["cohort_IQR"].fillna(0)
    q25 = scored["cohort_q25"]
    q75 = scored["cohort_q75"]
    fm  = scored["frac_meth"]

    scored["IQR_outlier"] = (
        fm.notna() & q25.notna() & q75.notna() & (
            (fm < q25 - 1.5 * iqr) | (fm > q75 + 1.5 * iqr)
        )
    )
    scored["Z_outlier"] = scored["z_score"].abs() >= z_thresh
    scored["outlier"]   = scored["IQR_outlier"] | scored["Z_outlier"]

    n_out = int(scored["outlier"].sum())
    n_iqr = int(scored["IQR_outlier"].sum())
    n_z   = int(scored["Z_outlier"].sum())
    log.info(
        "  Outliers : %d total  (%d IQR, %d |Z|≥%.1f, %d IQR∩Z)",
        n_out, n_iqr, n_z, z_thresh,
        int((scored["IQR_outlier"] & scored["Z_outlier"]).sum()),
    )
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# 5. Écriture des sorties
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    scored: pd.DataFrame,
    stats: pd.DataFrame,
    outdir: str,
) -> None:
    """Écrit les 3 fichiers TSV de sortie."""
    os.makedirs(outdir, exist_ok=True)

    base_cols = [
        "sample", "island_id", "island_name", "chrom",
        "island_start", "island_end", "island_cpgNum",
        "haplotype", "n_reads", "n_cpg_total", "n_cpg_meth", "n_cpg_unmeth", "frac_meth",
    ]
    stat_cols = [
        "cohort_n_samples", "cohort_mean", "cohort_std",
        "cohort_median", "cohort_q25", "cohort_q75", "cohort_IQR",
        "z_score", "p_value", "p_adj",
        "IQR_outlier", "Z_outlier", "outlier",
    ]
    out_cols  = base_cols + [c for c in stat_cols if c in scored.columns]
    out_cols += [c for c in scored.columns if c not in out_cols]

    p1 = os.path.join(outdir, "all_samples_stats.tsv")
    scored[out_cols].to_csv(p1, sep="\t", index=False)
    log.info("Fichier complet    → %s  (%d lignes)", p1, len(scored))

    p2 = os.path.join(outdir, "cohort_locus_stats.tsv")
    stats.to_csv(p2, sep="\t", index=False)
    log.info("Stats locus        → %s  (%d loci × haplotypes)", p2, len(stats))

    out_df = scored[scored["outlier"] == True][out_cols].copy()
    out_df = out_df.sort_values(
        ["island_id", "haplotype", "z_score"],
        key=lambda s: s.abs() if s.name == "z_score" else s,
        ascending=[True, True, False],
    )
    p3 = os.path.join(outdir, "outliers.tsv")
    out_df.to_csv(p3, sep="\t", index=False)
    log.info("Outliers           → %s  (%d entrées, %d loci, %d échantillons)",
             p3, len(out_df),
             out_df["island_id"].nunique(),
             out_df["sample"].nunique())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--counts", required=True,
                        help="Fichier TSV consolidé (all_samples_counts.tsv)")
    parser.add_argument("--outdir", required=True,
                        help="Répertoire de sortie")
    parser.add_argument("--exclude_samples", default=None,
                        help="Fichier texte listant les échantillons à exclure "
                             "(généré par 2_QC.py, une ligne par sample, # = commentaire)")
    parser.add_argument("--min_reads",     type=int,   default=15,
                        help="Profondeur minimale pour haplotype 'all' (défaut : 15)")
    parser.add_argument("--min_reads_hap", type=int,   default=8,
                        help="Profondeur minimale pour hap1/hap2/unphased (défaut : 8)")
    parser.add_argument("--min_cpg",       type=int,   default=10,
                        help="Nombre minimal de CpG annotés dans l'îlot (défaut : 10)")
    parser.add_argument("--min_samples",   type=int,   default=5,
                        help="Nombre minimal d'échantillons valides par locus (défaut : 5)")
    parser.add_argument("--z_thresh",      type=float, default=2.5,
                        help="Seuil |Z| pour flag Z_outlier (défaut : 2.5)")
    parser.add_argument("--leave_one_out", action="store_true",
                        help="Calculer les stats cohorte en excluant l'individu testé "
                             "(recommandé pour cohortes < 30 ; plus lent)")
    parser.add_argument("--force", action="store_true",
                        help="Recalculer même si les fichiers de sortie existent déjà")
    args = parser.parse_args()

    out_main = os.path.join(args.outdir, "all_samples_stats.tsv")
    if not args.force and os.path.isfile(out_main):
        log.info("Sortie déjà existante : %s\nUtilisez --force pour recalculer.", out_main)
        sys.exit(0)

    if not os.path.isfile(args.counts):
        sys.exit(f"[ERROR] Fichier introuvable : {args.counts}")

    # Lecture du fichier d'exclusion
    exclude_samples: list[str] = []
    if args.exclude_samples:
        exclude_samples = read_exclude_list(args.exclude_samples)

    log.info("=" * 60)
    log.info("  Analyse statistique cohorte – méthylation CpG")
    log.info("=" * 60)
    log.info("  Counts TSV        : %s", args.counts)
    log.info("  Outdir            : %s", args.outdir)
    log.info("  Exclusion samples : %s (%d échantillons)",
             args.exclude_samples or "(aucun)", len(exclude_samples))
    log.info("  min_reads         : %d  (all) / %d  (hap)", args.min_reads, args.min_reads_hap)
    log.info("  min_cpg           : %d", args.min_cpg)
    log.info("  min_samples       : %d", args.min_samples)
    log.info("  z_thresh          : %.1f", args.z_thresh)
    log.info("  leave_one_out     : %s", "oui" if args.leave_one_out else "non")
    log.info("=" * 60)

    # ── Chargement + filtre qualité + exclusion ───────────────────────────────
    filt = load_and_filter(
        args.counts,
        min_reads       = args.min_reads,
        min_reads_hap   = args.min_reads_hap,
        min_cpg         = args.min_cpg,
        exclude_samples = exclude_samples if exclude_samples else None,
    )

    # ── Stats cohorte par locus × haplotype ───────────────────────────────────
    stats = compute_cohort_stats(filt, min_samples=args.min_samples)

    # ── Z-score ───────────────────────────────────────────────────────────────
    if args.leave_one_out:
        scored = score_leave_one_out(filt, min_samples=args.min_samples)
    else:
        scored = score_standard(filt, stats)

    # ── P-value + FDR + flags ─────────────────────────────────────────────────
    scored = add_pvalue_and_flags(scored, z_thresh=args.z_thresh)

    # ── Sorties ───────────────────────────────────────────────────────────────
    write_outputs(scored, stats, args.outdir)

    log.info("=" * 60)
    log.info("Terminé. Résultats dans : %s", args.outdir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
