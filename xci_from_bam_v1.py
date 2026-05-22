#!/usr/bin/env python3
"""
Analyse XCI haplotype-spécifique depuis un BAM haplotaggé (PacBio HiFi)
------------------------------------------------------------------------
Pour chaque îlot CpG non pseudoautosomal du chrX :
  1. Sélectionne les reads couvrant intégralement l'île.
  2. Classe chaque read (méthylé / non méthylé / partiel) selon la fraction
     de CpG méthylés (tags MM/ML).
  3. Agrège par île et par haplotype (tag HP mis par WhatsHap haplotag).

Sorties :
  - <PREFIX>.chrx_xci.per_read.tsv       : une ligne par read retenu
  - <PREFIX>.chrx_xci.per_island.tsv     : une ligne par îlot (agrégat)
  - <PREFIX>.chrx_xci.scatter.html       : scatter méthylation hap1 vs hap2
  - <PREFIX>.chrx_xci.distribution.html  : distribution KDE par haplotype
  - <PREFIX>.chrx_xci.histogram.html     : histogrammes de méthylation hap1 / hap2
  - <PREFIX>.chrx_xci.igv.bed            : îlots colorés pour IGV/UCSC (BED9)
  - <PREFIX>.chrx_xci.summary.json       : stats clés (fusionnable multi-samples)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pysam
from scipy.stats import gaussian_kde, norm as sp_norm
from sklearn.mixture import GaussianMixture

# ------------------------------------------------------------------
# Régions pseudoautosomales de chrX (GRCh38) — exclues de l'analyse
# ------------------------------------------------------------------
PAR_REGIONS = [
    (60_001,       2_699_520),    # PAR1
    (154_931_044, 155_260_560),   # PAR2
]

# Seuils par défaut
DEFAULT_METH_PROB_THR     = 128    # ML ≥ 128/255 (≈ 0.5) → CpG méthylé
DEFAULT_MIN_CPG           = 10     # CpG minimum dans l'île pour retenir un read
DEFAULT_METH_FRAC_THR     = 0.75   # fraction min pour classification méthylé/non méthylé
DEFAULT_MIN_READS_PER_HAP = 8      # reads informatifs (méthylés + non méthylés) minimum par haplotype
DEFAULT_HEMI_LOW          = 0.40   # borne basse hémi-méthylation (tous reads confondus)
DEFAULT_HEMI_HIGH         = 0.60   # borne haute hémi-méthylation (tous reads confondus)

CHROM = "chrX"


# ============================================================
# Arguments CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    _ex_cpgi = os.path.join(".", "cpg_island_chrX.bed")

    parser = argparse.ArgumentParser(
        description="Analyse XCI haplotype-spécifique depuis un BAM haplotaggé (PacBio HiFi).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemple :\n"
            f"  python {os.path.basename(__file__)} \\\n"
            f"      sample.GRCh38.haplotagged.bam \\\n"
            f"      --cpg-islands {_ex_cpgi} \\\n"
            f"      --prefix 21-04669 \\\n"
            f"      --out-dir ."
        ),
    )
    parser.add_argument(
        "bam",
        help="BAM haplotaggé (trié + indexé .bai, avec tags HP et MM/ML).",
    )
    parser.add_argument(
        "--cpg-islands", default=_ex_cpgi, metavar="BED",
        help=f"Fichier BED des îlots CpG sur chrX (≥ 3 colonnes, col 4 = nom optionnel). "
             f"(défaut : {_ex_cpgi})",
    )
    parser.add_argument(
        "--prefix", default="sample", metavar="PREFIX",
        help="Préfixe des fichiers de sortie. (défaut : sample)",
    )
    parser.add_argument(
        "--out-dir", default=".", metavar="DIR",
        help="Répertoire de sortie. (défaut : .)",
    )
    parser.add_argument(
        "--min-cpg", type=int, default=DEFAULT_MIN_CPG, metavar="N",
        help=f"Nombre minimum de CpG couverts dans l'île pour retenir un read. "
             f"(défaut : {DEFAULT_MIN_CPG})",
    )
    parser.add_argument(
        "--meth-frac-thr", type=float, default=DEFAULT_METH_FRAC_THR, metavar="F",
        help=f"Fraction min de CpG méthylés/non méthylés pour la classification. "
             f"(défaut : {DEFAULT_METH_FRAC_THR})",
    )
    parser.add_argument(
        "--meth-prob-thr", type=int, default=DEFAULT_METH_PROB_THR, metavar="N",
        help=f"Seuil ML (0-255) pour qu'un CpG soit considéré méthylé. "
             f"(défaut : {DEFAULT_METH_PROB_THR} ≈ 0.5)",
    )
    parser.add_argument(
        "--min-reads-per-hap", type=int, default=DEFAULT_MIN_READS_PER_HAP, metavar="N",
        help=f"Nombre minimum de reads informatifs (méthylés + non méthylés, hors partiels) "
             f"par haplotype pour retenir un îlot dans les figures. "
             f"(défaut : {DEFAULT_MIN_READS_PER_HAP})",
    )
    parser.add_argument(
        "--hemi-low", type=float, default=DEFAULT_HEMI_LOW, metavar="F",
        help=f"Borne basse de la fenêtre hémi-méthylation (fraction, tous reads confondus). "
             f"(défaut : {DEFAULT_HEMI_LOW})",
    )
    parser.add_argument(
        "--hemi-high", type=float, default=DEFAULT_HEMI_HIGH, metavar="F",
        help=f"Borne haute de la fenêtre hémi-méthylation (fraction, tous reads confondus). "
             f"(défaut : {DEFAULT_HEMI_HIGH})",
    )
    parser.add_argument(
        "--phasing-mode", required=True, choices=["read_backed", "pedigree"],
        metavar="MODE",
        help=(
            "Mode de phasage (obligatoire) : "
            "'read_backed' = phasage local par read, distribution bimodale par haplotype, "
            "biais estimé par GMM à 2 composantes ; "
            "'pedigree' = phasage chromosomique consistant, distribution unimodale par haplotype, "
            "biais estimé directement par la médiane."
        ),
    )
    return parser.parse_args()


# ============================================================
# Chargement des îlots CpG
# ============================================================

def load_cpg_islands(filepath: str, chrom: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit un fichier BED (≥ 3 colonnes) d'îlots CpG.
    Filtre sur `chrom` et exclut les régions PAR (GRCh38).
    La colonne 4 (nom) est utilisée comme island_id si présente,
    sinon génère un identifiant chrom:start-end.
    """
    df = pd.read_csv(filepath, sep="\t", comment="#", header=None)
    # Garde au maximum 4 colonnes
    df = df.iloc[:, : min(4, df.shape[1])].copy()
    col_names = ["chrom", "start", "end", "name"][: df.shape[1]]
    df.columns = col_names
    if "name" not in df.columns:
        df["name"] = pd.NA

    df = df[df["chrom"] == chrom].copy()
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)

    # Remplir les noms manquants
    no_name = df["name"].isna() | (df["name"].astype(str).str.strip() == "")
    df.loc[no_name, "name"] = (
        df.loc[no_name, "chrom"] + ":"
        + df.loc[no_name, "start"].astype(str) + "-"
        + df.loc[no_name, "end"].astype(str)
    )

    # Exclure les régions PAR (chevauchement partiel inclus)
    par_mask = pd.Series(False, index=df.index)
    for par_start, par_end in PAR_REGIONS:
        par_mask |= (df["start"] < par_end) & (df["end"] > par_start)
    par_df = df[par_mask].reset_index(drop=True)
    df     = df[~par_mask].reset_index(drop=True)
    if len(par_df):
        print(f"  [INFO] {len(par_df)} îlot(s) PAR exclus")

    return df, par_df


# ============================================================
# Parsing méthylation par read
# ============================================================

def get_cpg_methylation_in_island(
    read: pysam.AlignedSegment,
    island_start: int,
    island_end: int,
    meth_prob_thr: int,
) -> tuple | None:
    """
    Retourne (n_cpg, n_meth, n_unmeth) pour les positions 5mC (tag MM/ML)
    du read qui tombent dans [island_start, island_end).
    Retourne None si le read n'a pas de tags de méthylation 5mC, ou si
    aucun CpG couvert dans l'île.
    """
    try:
        mod_bases = read.modified_bases  # {(base, strand, mod): {qpos: prob 0-255}}
    except (AttributeError, ValueError, KeyError):
        return None

    # Collecte toutes les positions 5mC (tous brins)
    meth_by_qpos: dict = {}
    for (base, _strand, mod), qpos_dict in mod_bases.items():
        if base == "C" and mod == "m":
            meth_by_qpos.update(qpos_dict)

    if not meth_by_qpos:
        return None

    # Mapping query_pos → ref_pos (bases alignées uniquement)
    qpos_to_rpos: dict = dict(read.get_aligned_pairs(matches_only=True))

    n_meth = n_unmeth = 0
    for qpos, prob in meth_by_qpos.items():
        rpos = qpos_to_rpos.get(qpos)
        if rpos is None:
            continue
        if island_start <= rpos < island_end:
            if prob >= meth_prob_thr:
                n_meth += 1
            else:
                n_unmeth += 1

    n_cpg = n_meth + n_unmeth
    if n_cpg == 0:
        return None
    return n_cpg, n_meth, n_unmeth


def classify_read(n_meth: int, n_cpg: int, meth_frac_thr: float) -> str:
    """Classe un read en méthylé / non méthylé / partiel."""
    frac = n_meth / n_cpg
    if frac >= meth_frac_thr:
        return "methylated"
    if frac <= (1.0 - meth_frac_thr):
        return "unmethylated"
    return "partial"


def _delta_to_rgb(delta: float) -> str:
    """
    Convertit un delta de méthylation (meth_frac_hap1 − meth_frac_hap2)
    en couleur RGB pour BED9 IGV/UCSC.
      delta > 0 (hap1 plus méthylé) → bleu
      delta < 0 (hap2 plus méthylé) → rouge
    Intensité proportionnelle à |delta|, normalisée sur 0.5.
    """
    t      = min(abs(delta) / 0.5, 1.0)
    n      = (210, 210, 210)   # neutre
    target = (10, 80, 200) if delta >= 0 else (200, 50, 10)
    r = int(n[0] + t * (target[0] - n[0]))
    g = int(n[1] + t * (target[1] - n[1]))
    b = int(n[2] + t * (target[2] - n[2]))
    return f"{r},{g},{b}"

SEX_CHRY_THRESHOLD = 0.10   # fraction chrY/(chrX+chrY) au-delà de laquelle → M
CHROM_Y            = "chrY"


def _infer_sex(bam_path: str) -> tuple[str, int, int]:
    """
    Infère le sexe chromosomique en lisant les statistiques de l'index BAI
    (nombre de reads mappés par contig, sans parcourir les reads).
    Retourne (sexe, n_chrx, n_chry).
    """
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        stats  = {s.contig: s.mapped for s in bam.get_index_statistics()}
    n_chrx = stats.get(CHROM,   0)
    n_chry = stats.get(CHROM_Y, 0)
    total  = n_chrx + n_chry
    if total == 0:
        return "unknown", 0, 0
    chry_frac = n_chry / total
    return ("M" if chry_frac > SEX_CHRY_THRESHOLD else "F"), n_chrx, n_chry


def _fit_gmm(pct_series: pd.Series, init_low: float = 25.0, init_high: float = 75.0):
    """
    Ajuste un GMM à 2 composantes gaussiennes sur une série de pourcentages (0–100).
    Initialisation aux positions init_low / init_high pour stabiliser la convergence.
    Retourne (means, stds, weights) triés par moyenne croissante, ou None si < 10 valeurs.
    """
    vals = pct_series.dropna().values
    if len(vals) < 10:
        return None
    gmm = GaussianMixture(
        n_components=2,
        means_init=[[init_low], [init_high]],
        random_state=42,
        max_iter=300,
    )
    gmm.fit(vals.reshape(-1, 1))
    order   = np.argsort(gmm.means_.ravel())
    means   = gmm.means_.ravel()[order]
    stds    = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]
    return means, stds, weights


def main() -> None:
    args = parse_args()

    bam_path  = os.path.realpath(args.bam)
    out_dir   = os.path.realpath(args.out_dir)
    PREFIX    = args.prefix
    MIN_CPG   = args.min_cpg
    METH_FRAC     = args.meth_frac_thr
    METH_PROB     = args.meth_prob_thr
    MIN_READS_HAP = args.min_reads_per_hap
    HEMI_LOW      = args.hemi_low
    HEMI_HIGH     = args.hemi_high
    PHASING_MODE  = args.phasing_mode

    os.makedirs(out_dir, exist_ok=True)

    OUT_PER_READ   = os.path.join(out_dir, f"{PREFIX}.chrx_xci.per_read.tsv")
    OUT_PER_ISLAND = os.path.join(out_dir, f"{PREFIX}.chrx_xci.per_island.tsv")
    OUT_SCATTER    = os.path.join(out_dir, f"{PREFIX}.chrx_xci.scatter.html")
    OUT_DISTRIB    = os.path.join(out_dir, f"{PREFIX}.chrx_xci.distribution.html")
    OUT_HIST       = os.path.join(out_dir, f"{PREFIX}.chrx_xci.histogram.html")
    OUT_BED        = os.path.join(out_dir, f"{PREFIX}.chrx_xci.igv.bed")
    OUT_SUMMARY    = os.path.join(out_dir, f"{PREFIX}.chrx_xci.summary.json")

    print(f"BAM          : {bam_path}")
    print(f"Îlots CpG    : {args.cpg_islands}")
    print(f"Sorties      : {out_dir}")
    print(f"Paramètres   : min_cpg={MIN_CPG}, meth_frac_thr={METH_FRAC}, "
          f"meth_prob_thr={METH_PROB}/255, min_reads_per_hap={MIN_READS_HAP}, "
          f"hemi=[{HEMI_LOW:.0%}–{HEMI_HIGH:.0%}], phasing_mode={PHASING_MODE}\n")

    # Vérification de l'index BAM
    bai_candidates = [bam_path + ".bai", bam_path.replace(".bam", ".bai")]
    if not any(os.path.exists(p) for p in bai_candidates):
        print("[WARN] Index BAM (.bai) introuvable. "
              "Assurez-vous que le BAM est indexé (samtools index).",
              file=sys.stderr)

    # ------------------------------------------------------------------
    # 0. Sex-check rapide (indépendant de l'analyse XCI)
    # ------------------------------------------------------------------
    print("[0/7] Inférence du sexe via les statistiques de l'index BAI...")
    inferred_sex, _sex_n_chrx, _sex_n_chry = _infer_sex(bam_path)
    _sex_total     = _sex_n_chrx + _sex_n_chry
    _sex_chry_frac = (_sex_n_chry / _sex_total) if _sex_total > 0 else float("nan")
    print(f"      chrX={_sex_n_chrx}, chrY={_sex_n_chry}, "
          f"frac_chrY={_sex_chry_frac:.0%}  →  sexe inféré : {inferred_sex}")

    if inferred_sex == "M":
        print("      [INFO] Sexe masculin : analyse XCI non applicable. "
              "Production du résumé JSON uniquement.")
        summary_early = {
            "sample":                   PREFIX,
            "bam":                      bam_path,
            "inferred_sex":             inferred_sex,
            "sex_check_n_chrx_sampled": _sex_n_chrx,
            "sex_check_n_chry_sampled": _sex_n_chry,
            "sex_check_chry_frac":      round(_sex_chry_frac, 3) if not np.isnan(_sex_chry_frac) else None,
            "phasing_mode":             PHASING_MODE,
            "params": {
                "min_cpg":           MIN_CPG,
                "meth_frac_thr":     METH_FRAC,
                "meth_prob_thr":     METH_PROB,
                "min_reads_per_hap": MIN_READS_HAP,
                "hemi_low":          HEMI_LOW,
                "hemi_high":         HEMI_HIGH,
            },
        }
        with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
            json.dump(summary_early, fh, indent=2, ensure_ascii=False)
        print(f"      Résumé JSON  → {OUT_SUMMARY}")
        print("[--] Analyse XCI ignorée (sexe M).")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 1. Chargement des îlots CpG (non PAR)
    # ------------------------------------------------------------------
    print(f"[1/7] Chargement des îlots CpG non pseudoautosomaux ({CHROM})...")
    islands_df, par_islands_df = load_cpg_islands(args.cpg_islands, chrom=CHROM)
    print(f"      Îlots retenus : {len(islands_df)}")

    if islands_df.empty:
        print("[ERREUR] Aucun îlot CpG disponible après filtrage.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Parcours du BAM — classification de chaque read par île
    # ------------------------------------------------------------------
    print(f"[2/7] Analyse des reads par îlot CpG ({len(islands_df)} îlots)...")
    per_read_records = []
    n_islands = len(islands_df)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for idx, island in islands_df.iterrows():
            if (idx + 1) % 100 == 0 or (idx + 1) == n_islands:
                print(f"      Progression : {idx + 1}/{n_islands} îlots", end="\r", flush=True)

            istart = int(island["start"])
            iend   = int(island["end"])
            iname  = str(island["name"])

            for read in bam.fetch(CHROM, istart, iend):
                # Filtres de base
                if (read.is_unmapped or read.is_secondary
                        or read.is_supplementary or read.is_duplicate):
                    continue

                # Overlap complet : le read doit couvrir intégralement l'île
                if read.reference_start > istart or read.reference_end < iend:
                    continue

                # Haplotype (tag HP : 1, 2 ou absent → 0 = non phasé)
                hp = int(read.get_tag("HP")) if read.has_tag("HP") else 0

                # Méthylation CpG dans l'île
                meth_data = get_cpg_methylation_in_island(read, istart, iend, METH_PROB)
                if meth_data is None:
                    continue
                n_cpg, n_meth, n_unmeth = meth_data

                if n_cpg < MIN_CPG:
                    continue

                classification = classify_read(n_meth, n_cpg, METH_FRAC)

                per_read_records.append({
                    "read_id":        read.query_name,
                    "haplotype":      hp,
                    "island_id":      iname,
                    "island_chrom":   CHROM,
                    "island_start":   istart,
                    "island_end":     iend,
                    "n_cpg":          n_cpg,
                    "n_cpg_meth":     n_meth,
                    "n_cpg_unmeth":   n_unmeth,
                    "meth_fraction":  round(n_meth / n_cpg, 4),
                    "classification": classification,
                })

    print()  # saut de ligne après \r
    per_read_df = pd.DataFrame(per_read_records)
    print(f"      Reads retenus : {len(per_read_df)}")

    if per_read_df.empty:
        print("[ERREUR] Aucun read ne passe les filtres.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Écriture de la table par read + agrégation par île
    # ------------------------------------------------------------------
    print("[3/7] Écriture des tables...")
    per_read_df.to_csv(OUT_PER_READ, sep="\t", index=False)
    print(f"      Par read     → {OUT_PER_READ}")

    # Agrégation par île
    def _agg_island(grp):
        row = {}
        for hp_label, hp_val in [("hap1", 1), ("hap2", 2), ("unphased", 0)]:
            sub  = grp[grp["haplotype"] == hp_val]
            n    = len(sub)
            nm   = int((sub["classification"] == "methylated").sum())
            nu   = int((sub["classification"] == "unmethylated").sum())
            np_  = int((sub["classification"] == "partial").sum())
            denom = nm + nu
            row[f"n_reads_{hp_label}"]   = n
            row[f"n_meth_{hp_label}"]    = nm
            row[f"n_unmeth_{hp_label}"]  = nu
            row[f"n_partial_{hp_label}"] = np_
            row[f"meth_frac_{hp_label}"] = round(nm / denom, 4) if denom > 0 else float("nan")
        return pd.Series(row)

    per_island_df = (
        per_read_df
        .groupby(["island_id", "island_chrom", "island_start", "island_end"])
        .apply(_agg_island)
        .reset_index()
    )
    per_island_df["island_length"] = (
        per_island_df["island_end"] - per_island_df["island_start"]
    )
    per_island_df["delta_meth"] = (
        per_island_df["meth_frac_hap1"] - per_island_df["meth_frac_hap2"]
    ).round(4)

    # La fraction globale = (n_meth_hap1 + n_meth_hap2) / (n_informatifs_hap1 + n_informatifs_hap2)
    # (reads partiels exclus du numérateur et du dénominateur)
    per_island_df["n_informative_total"] = (
        per_island_df["n_meth_hap1"]   + per_island_df["n_unmeth_hap1"] +
        per_island_df["n_meth_hap2"]   + per_island_df["n_unmeth_hap2"]
    )
    per_island_df["n_meth_total"] = (
        per_island_df["n_meth_hap1"] + per_island_df["n_meth_hap2"]
    )
    per_island_df["meth_frac_combined"] = (
        per_island_df["n_meth_total"] / per_island_df["n_informative_total"]
    ).where(per_island_df["n_informative_total"] > 0).round(4)

    per_island_df.to_csv(OUT_PER_ISLAND, sep="\t", index=False)
    print(f"      Par île      → {OUT_PER_ISLAND}")

    # ------------------------------------------------------------------
    # 4. Fichier BED de visualisation (IGV/UCSC)
    # ------------------------------------------------------------------
    print("[4/7] Écriture du fichier BED de visualisation (IGV/UCSC)...")

    # Îles PAR
    bed_rows = []
    for _, row in par_islands_df.iterrows():
        s = int(row["start"]); e = int(row["end"]); nm = str(row["name"])
        bed_rows.append((CHROM, s, e, f"{nm}|PAR", 0, ".", s, e, "128,128,128"))

    # Fusion îles non-PAR avec les données agrégées — jointure sur nom + coordonnées
    # pour éviter les doublons en cas de noms non uniques dans le fichier BED source
    islands_info = islands_df.merge(
        per_island_df[[
            "island_id", "island_start", "island_end",
            "n_meth_hap1", "n_unmeth_hap1",
            "n_meth_hap2", "n_unmeth_hap2",
            "meth_frac_combined", "delta_meth",
            "meth_frac_hap1", "meth_frac_hap2",
        ]],
        left_on=["name", "start", "end"],
        right_on=["island_id", "island_start", "island_end"],
        how="left",
    )
    for _, row in islands_info.iterrows():
        s    = int(row["start"])
        e    = int(row["end"])
        nm   = str(row["name"])
        comb = row.get("meth_frac_combined")
        n_h1 = (row.get("n_meth_hap1", 0) or 0) + (row.get("n_unmeth_hap1", 0) or 0)
        n_h2 = (row.get("n_meth_hap2", 0) or 0) + (row.get("n_unmeth_hap2", 0) or 0)
        if pd.isna(comb):
            status = "no_data"
            rgb    = "200,200,200"
            score  = 0
        elif int(n_h1) < MIN_READS_HAP or int(n_h2) < MIN_READS_HAP:
            status = f"low_cov(h1={int(n_h1)},h2={int(n_h2)})"
            rgb    = "170,170,170"
            score  = 0
        elif float(comb) < HEMI_LOW:
            status = f"unmethylated(comb={float(comb):.0%})"
            rgb    = "140,190,230"
            score  = 0
        elif float(comb) > HEMI_HIGH:
            status = f"methylated(comb={float(comb):.0%})"
            rgb    = "230,160,100"
            score  = 0
        else:
            delta  = float(row["delta_meth"])
            h1v    = float(row["meth_frac_hap1"])
            h2v    = float(row["meth_frac_hap2"])
            status = f"d={delta:+.2f}(h1={h1v:.0%},h2={h2v:.0%})"
            rgb    = _delta_to_rgb(delta)
            score  = min(int(abs(delta) * 1000), 1000)
        bed_rows.append((CHROM, s, e, f"{nm}|{status}", score, ".", s, e, rgb))

    bed_rows.sort(key=lambda r: r[1])
    with open(OUT_BED, "w") as fh:
        fh.write(
            f'track name="{PREFIX}_chrX_XCI" '
            f'description="CpG island methylation bias {CHROM} \u2014 {PREFIX}" '
            f'itemRgb="On"\n'
        )
        for r in bed_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"      BED IGV/UCSC → {OUT_BED}")

    # ------------------------------------------------------------------
    # 5. Figures Plotly
    # ------------------------------------------------------------------
    print("[5/7] Génération des figures...")

    n_cov = int((
        (per_island_df["n_meth_hap1"] + per_island_df["n_unmeth_hap1"] >= MIN_READS_HAP) &
        (per_island_df["n_meth_hap2"] + per_island_df["n_unmeth_hap2"] >= MIN_READS_HAP)
    ).sum())
    valid = per_island_df[
        (per_island_df["n_meth_hap1"] + per_island_df["n_unmeth_hap1"] >= MIN_READS_HAP) &
        (per_island_df["n_meth_hap2"] + per_island_df["n_unmeth_hap2"] >= MIN_READS_HAP) &
        (per_island_df["meth_frac_combined"] >= HEMI_LOW) &
        (per_island_df["meth_frac_combined"] <= HEMI_HIGH)
    ].copy()
    print(f"      Ìlots avec ≥ {MIN_READS_HAP} reads informatifs/haplotype : {n_cov} / {len(per_island_df)}")
    print(f"      dont hémi-méthylés ({HEMI_LOW:.0%}–{HEMI_HIGH:.0%}) : {len(valid)}")
    h1_pct  = valid["meth_frac_hap1"] * 100
    h2_pct  = valid["meth_frac_hap2"] * 100
    h1_med  = h1_pct.median() if not h1_pct.empty else float("nan")
    h2_med  = h2_pct.median() if not h2_pct.empty else float("nan")
    h1_mean = h1_pct.mean()   if not h1_pct.empty else float("nan")
    h2_mean = h2_pct.mean()   if not h2_pct.empty else float("nan")

    # --- Estimation du biais XCI ---
    gmm_combined = None
    xci_bias = xci_bias_hap1 = xci_bias_hap2 = float("nan")
    if PHASING_MODE == "read_backed":
        # Série combinée : concaténation hap1+hap2 (chaque île contribue 2 observations)
        combined_pct = pd.concat([h1_pct, h2_pct], ignore_index=True)
        gmm_combined = _fit_gmm(combined_pct)
        if gmm_combined is not None:
            peak_low  = gmm_combined[0][0]
            peak_high = gmm_combined[0][1]
            xci_bias  = (peak_low / 100.0 + (1.0 - peak_high / 100.0)) / 2.0
            print(f"      GMM combin\u00e9 : \u03bc_low={peak_low:.1f}%  \u03bc_high={peak_high:.1f}%  "
                  f"\u2192 biais XCI = {xci_bias:.0%}\u2013{1 - xci_bias:.0%}")
    else:  # pedigree
        combined_pct = pd.concat([h1_pct, h2_pct], ignore_index=True)
        xci_bias_hap1 = float(h1_med / 100) if not np.isnan(h1_med) else float("nan")
        xci_bias_hap2 = float(h2_med / 100) if not np.isnan(h2_med) else float("nan")
        print(f"      Hap1 m\u00e9diane : {h1_med:.1f}%  \u2192 fraction X inactif hap1 = {xci_bias_hap1:.0%}")
        print(f"      Hap2 m\u00e9diane : {h2_med:.1f}%  \u2192 fraction X inactif hap2 = {xci_bias_hap2:.0%}")

    # --- HTML résumé biais (inséré dans le tableau de stats) ---
    if PHASING_MODE == "read_backed" and gmm_combined is not None:
        _bias_str = f"{xci_bias:.0%}\u2013{1 - xci_bias:.0%}" if not np.isnan(xci_bias) else "N/A"
        _bias_html = (
            f"<tr><td><strong>Biais XCI (GMM combin\u00e9)</strong></td>"
            f"<td colspan='2' style='text-align:center'><strong>{_bias_str}</strong></td></tr>"
            f"<tr><td><strong>Pic actif \u2013 \u03bc GMM</strong></td>"
            f"<td colspan='2' style='text-align:center'>{gmm_combined[0][0]:.1f}%</td></tr>"
            f"<tr><td><strong>Pic inactif \u2013 \u03bc GMM</strong></td>"
            f"<td colspan='2' style='text-align:center'>{gmm_combined[0][1]:.1f}%</td></tr>"
        )
    elif PHASING_MODE == "pedigree":
        _bh1 = f"{xci_bias_hap1:.0%}" if not np.isnan(xci_bias_hap1) else "N/A"
        _bh2 = f"{xci_bias_hap2:.0%}" if not np.isnan(xci_bias_hap2) else "N/A"
        _bias_html = (
            f"<tr><td><strong>Fraction X inactif (m\u00e9diane)</strong></td>"
            f"<td class='pat'>{_bh1}</td><td class='mat'>{_bh2}</td></tr>"
        )
    else:
        _bias_html = ""

    # --- Sections HTML communes aux deux figures ---
    _html_sections = f"""
<div class="section">
  <h3>Objectif</h3>
  <p>D\u00e9tection d'un biais d'inactivation du chromosome X (XCI) par analyse haplotype-sp\u00e9cifique
  de la m\u00e9thylation des \u00eelots CpG non pseudoautosomaux. Chaque read couvrant int\u00e9gralement
  un \u00eelot est classifi\u00e9 (m\u00e9thyl\u00e9 / non m\u00e9thyl\u00e9 / partiel) selon la fraction de CpG m\u00e9thyl\u00e9s,
  puis les r\u00e9sultats sont agr\u00e9g\u00e9s par haplotype (tag HP WhatsHap).</p>
</div>
<div class="section">
  <h3>R\u00e9sultats</h3>
  <p>
    \u00cclots non-PAR : <strong>{len(islands_df)}</strong>
    &nbsp;|&nbsp; Reads retenus : <strong>{len(per_read_df)}</strong>
    &nbsp;|&nbsp; Couverture \u2265{MIN_READS_HAP} reads/haplotype : <strong>{n_cov}</strong>
    &nbsp;|&nbsp; H\u00e9mi-m\u00e9thyl\u00e9s ({HEMI_LOW:.0%}\u2013{HEMI_HIGH:.0%}) retenus : <strong>{len(valid)}</strong>
  </p>
  <table class="stats">
    <tr>
      <td></td>
      <td class="pat">&#9632; Haplotype 1</td>
      <td class="mat">&#9632; Haplotype 2</td>
    </tr>
    <tr>
      <td><strong>M\u00e9diane m\u00e9thylation</strong></td>
      <td class="pat">{h1_med:.1f}%</td>
      <td class="mat">{h2_med:.1f}%</td>
    </tr>
    <tr>
      <td><strong>Moyenne m\u00e9thylation</strong></td>
      <td class="pat">{h1_mean:.1f}%</td>
      <td class="mat">{h2_mean:.1f}%</td>
    </tr>
    {_bias_html}
  </table>
</div>
<div class="section">
  <h3>M\u00e9thodes</h3>
  <p>
    <strong>Donn\u00e9es :</strong> BAM haplotaggu\u00e9 PacBio HiFi (tags HP, MM/ML)<br>
    <strong>R\u00e9gions :</strong> {CHROM} non pseudoautosomal (PAR1/PAR2 GRCh38 exclus)<br>
    <strong>Mode de phasage :</strong> {PHASING_MODE}<br>
    <strong>Seuils :</strong>
    CpG m\u00e9thyl\u00e9 si ML &ge;&nbsp;{METH_PROB}/255 (\u2248&nbsp;{METH_PROB / 255:.2f}),
    minimum {MIN_CPG} CpG/read dans l'\u00eele,
    classification m\u00e9thyl\u00e9/non m\u00e9thyl\u00e9 si fraction &ge;&nbsp;{METH_FRAC:.0%},
    minimum {MIN_READS_HAP} reads informatifs (m\u00e9thyl\u00e9s + non m\u00e9thyl\u00e9s) par haplotype,
    h\u00e9mi-m\u00e9thylation globale [{HEMI_LOW:.0%}\u2013{HEMI_HIGH:.0%}] (tous reads confondus, partiels exclus)
  </p>
</div>
"""

    _HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>{page_title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 980px;
      margin: 0 auto;
      padding: 16px 24px 40px;
      color: #333;
    }}
    .section {{ margin-bottom: 16px; }}
    .section h3 {{
      margin: 0 0 5px 0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #888;
      border-bottom: 1px solid #e0e0e0;
      padding-bottom: 3px;
    }}
    .section p {{ margin: 4px 0 0; font-size: 12px; line-height: 1.65; }}
    table.stats {{ border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
    table.stats td {{ padding: 3px 24px 3px 0; }}
    .pat {{ color: steelblue; font-weight: bold; }}
    .mat {{ color: tomato;    font-weight: bold; }}
  </style>
</head>
<body>
  <div>{fig_div}</div>
  <div class="sections">{sections}</div>
</body>
</html>
"""

    def _write_html_page(fig, title: str, out_path: str) -> None:
        fig_div = fig.to_html(full_html=False, include_plotlyjs="cdn")
        html = _HTML_TEMPLATE.format(
            page_title=title,
            fig_div=fig_div,
            sections=_html_sections,
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)

    # ---- Figure 1 : Scatter hap1 vs hap2 par île --------------------------
    abs_delta = (valid["meth_frac_hap1"] - valid["meth_frac_hap2"]).abs()

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=h1_pct,
        y=h2_pct,
        mode="markers",
        marker=dict(
            color=abs_delta,
            colorscale="RdBu_r",
            cmin=0, cmax=1,
            size=6,
            opacity=0.7,
            colorbar=dict(title="|Δ méth|"),
        ),
        text=valid["island_id"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Hap1 : %{x:.1f}%<br>"
            "Hap2 : %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    # Diagonale de référence (pas de biais)
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=100, y1=100,
        line=dict(color="gray", dash="dot", width=1),
    )
    fig_scatter.update_layout(
        title=dict(
            text=f"Méthylation par îlot CpG — hap1 vs hap2 — {CHROM} | {PREFIX}",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Méthylation hap1 (%)", range=[0, 100]),
        yaxis=dict(title="Méthylation hap2 (%)", range=[0, 100]),
        template="plotly_white",
        width=700, height=650,
        margin=dict(t=80, b=60),
    )
    _write_html_page(
        fig_scatter,
        title=f"XCI {CHROM} — {PREFIX} — Scatter hap1 vs hap2",
        out_path=OUT_SCATTER,
    )
    print(f"      Scatter      → {OUT_SCATTER}")

    # ---- Figure 2 : Distribution KDE hap1 / hap2 / combiné ---------------
    fig_dist = go.Figure()
    x_grid = np.linspace(0, 100, 500)

    for pct, color, fill_color, name in [
        (h1_pct,       "steelblue", "rgba(70,130,180,0.15)", "Haplotype 1"),
        (h2_pct,       "tomato",    "rgba(255,99,71,0.15)",  "Haplotype 2"),
        (combined_pct, "seagreen",  "rgba(46,139,87,0.10)",  "Combiné (hap1+hap2)"),
    ]:
        if len(pct.dropna()) >= 2:
            kde = gaussian_kde(pct.dropna(), bw_method="scott")
            fig_dist.add_trace(go.Scatter(
                x=x_grid, y=kde(x_grid),
                name=name, mode="lines",
                line=dict(color=color, width=3),
                fill="tozeroy", fillcolor=fill_color,
            ))

    # Overlay GMM combiné (read_backed) ou ligne médiane verticale (pedigree)
    if PHASING_MODE == "read_backed" and gmm_combined is not None:
        means_g, stds_g, weights_g = gmm_combined
        for k in range(2):
            y_comp = weights_g[k] * sp_norm.pdf(x_grid, means_g[k], stds_g[k])
            fig_dist.add_trace(go.Scatter(
                x=x_grid, y=y_comp,
                mode="lines",
                line=dict(color="seagreen", width=1.5, dash="dash"),
                showlegend=False,
                hoverinfo="none",
            ))
    elif PHASING_MODE == "pedigree":
        for pct_val, color in [(h1_med, "steelblue"), (h2_med, "tomato")]:
            if not np.isnan(pct_val):
                fig_dist.add_vline(
                    x=pct_val,
                    line=dict(color=color, dash="dash", width=1.5),
                )

    fig_dist.update_layout(
        title=dict(
            text=(
                f"Distribution de la méthylation par îlot — {CHROM} | {PREFIX}"
                " — KDE (noyau gaussien, bande passante de Scott)"
            ),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Fraction de méthylation des îlots (%)", range=[0, 100]),
        yaxis_title="Densité de probabilité",
        legend=dict(x=0.70, y=0.95),
        template="plotly_white",
        width=960, height=500,
        margin=dict(t=80, b=60),
    )
    _write_html_page(
        fig_dist,
        title=f"XCI {CHROM} — {PREFIX} — Distribution",
        out_path=OUT_DISTRIB,
    )
    print(f"      Distribution → {OUT_DISTRIB}")

    # ---- Figure 3 : Histogrammes hap1 / hap2 / combiné (counts) ----------
    fig_hist = go.Figure()
    _bin_width = 100.0 / 50  # 50 bins sur [0, 100]
    for pct, color, opacity, name in [
        (h1_pct,       "steelblue", 0.55, "Haplotype 1"),
        (h2_pct,       "tomato",    0.55, "Haplotype 2"),
        (combined_pct, "seagreen",  0.40, "Combiné (hap1+hap2)"),
    ]:
        fig_hist.add_trace(go.Histogram(
            x=pct.dropna(),
            name=name,
            nbinsx=50,
            marker_color=color,
            opacity=opacity,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Méthylation : %{x:.1f}%<br>"
                "Nombre d'\u00eelots : %{y}<br>"
                "<extra></extra>"
            ),
        ))
    # Overlay GMM combiné (read_backed) ou ligne médiane verticale (pedigree)
    if PHASING_MODE == "read_backed" and gmm_combined is not None:
        means_g, stds_g, weights_g = gmm_combined
        n_combined = len(combined_pct.dropna())
        for k in range(2):
            y_comp = n_combined * _bin_width * weights_g[k] * sp_norm.pdf(x_grid, means_g[k], stds_g[k])
            fig_hist.add_trace(go.Scatter(
                x=x_grid, y=y_comp,
                mode="lines",
                line=dict(color="seagreen", width=1.5, dash="dash"),
                showlegend=False,
                hoverinfo="none",
            ))
    elif PHASING_MODE == "pedigree":
        for pct_val, color in [(h1_med, "steelblue"), (h2_med, "tomato")]:
            if not np.isnan(pct_val):
                fig_hist.add_vline(
                    x=pct_val,
                    line=dict(color=color, dash="dash", width=1.5),
                )

    fig_hist.update_layout(
        barmode="overlay",
        title=dict(
            text=(
                f"Distribution de la méthylation par îlot — {CHROM} | {PREFIX}"
                " — Histogrammes (counts, 50 bins)"
            ),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Fraction de méthylation des îlots (%)", range=[0, 100]),
        yaxis_title="Nombre d'îlots",
        legend=dict(x=0.70, y=0.95),
        template="plotly_white",
        width=960, height=500,
        margin=dict(t=80, b=60),
    )
    _write_html_page(
        fig_hist,
        title=f"XCI {CHROM} — {PREFIX} — Histogrammes",
        out_path=OUT_HIST,
    )
    print(f"      Histogrammes → {OUT_HIST}")

    # ------------------------------------------------------------------
    # 6. Résumé JSON (fusionnable multi-samples)
    # ------------------------------------------------------------------
    print("[6/7] Écriture du résumé JSON...")

    summary = {
        "sample":                       PREFIX,
        "bam":                          bam_path,
        "inferred_sex":                 inferred_sex,
        "sex_check_n_chrx_sampled":     _sex_n_chrx,
        "sex_check_n_chry_sampled":     _sex_n_chry,
        "sex_check_chry_frac":          round(_sex_chry_frac, 3) if not np.isnan(_sex_chry_frac) else None,
        "n_islands_nonPAR":             len(islands_df),
        "n_reads_retained":             len(per_read_df),
        "n_islands_sufficient_coverage": n_cov,
        "n_islands_hemi_methylated":    len(valid),
        "meth_frac_hap1_median":        round(float(h1_med),  4) if not np.isnan(h1_med)  else None,
        "meth_frac_hap2_median":        round(float(h2_med),  4) if not np.isnan(h2_med)  else None,
        "meth_frac_hap1_mean":          round(float(h1_mean), 4) if not np.isnan(h1_mean) else None,
        "meth_frac_hap2_mean":          round(float(h2_mean), 4) if not np.isnan(h2_mean) else None,
        "phasing_mode":                 PHASING_MODE,
        **({
            "xci_bias":      round(xci_bias, 4) if not np.isnan(xci_bias) else None,
            "gmm_peak_low":  round(float(gmm_combined[0][0]), 2) if gmm_combined is not None else None,
            "gmm_peak_high": round(float(gmm_combined[0][1]), 2) if gmm_combined is not None else None,
        } if PHASING_MODE == "read_backed" else {
            "xci_bias_hap1": round(xci_bias_hap1, 4) if not np.isnan(xci_bias_hap1) else None,
            "xci_bias_hap2": round(xci_bias_hap2, 4) if not np.isnan(xci_bias_hap2) else None,
        }),
        "params": {
            "min_cpg":          MIN_CPG,
            "meth_frac_thr":    METH_FRAC,
            "meth_prob_thr":    METH_PROB,
            "min_reads_per_hap": MIN_READS_HAP,
            "hemi_low":         HEMI_LOW,
            "hemi_high":        HEMI_HIGH,
        },
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"      Résumé JSON  → {OUT_SUMMARY}")

    print("[7/7] Terminé.")


if __name__ == "__main__":
    main()
