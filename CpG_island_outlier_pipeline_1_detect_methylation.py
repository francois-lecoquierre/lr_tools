#!/usr/bin/env python3
"""
1_detect_methylation.py – Extraction des comptes bruts de méthylation
=======================================================================
Pour chaque BAM, extrait les comptes de CpG méthylés / non méthylés par
îlot CpG et par haplotype, puis fusionne tout en un seul fichier TSV.

Colonnes de sortie (une ligne par échantillon × locus × haplotype) :
    sample          – nom de l'échantillon (stem du fichier BAM)
    island_id       – identifiant du locus  chrom:start-end
    island_name     – symbole du gène MANE ou nom UCSC
    chrom           – chromosome
    island_start    – début de l'îlot (0-based)
    island_end      – fin de l'îlot
    island_cpgNum   – nombre de CpG annotés dans l'îlot (référence UCSC)
    haplotype       – hap1 | hap2 | unphased | all
    n_reads         – nombre de lectures contribuant au locus
    n_cpg_total     – total de CpG observés (méthylés + non méthylés)
    n_cpg_meth      – CpG méthylés (prob ≥ cutoff)
    n_cpg_unmeth    – CpG non méthylés (prob < cutoff)
    frac_meth       – fraction de méthylation  n_cpg_meth / n_cpg_total

Usage :
    python 1_detect_methylation.py \\
        --bam_dir   <DIR_WITH_BAM_FILES> \\
        --cpg_tsv   <CPG_ISLANDS_TSV>    \\
        --outdir    <OUTPUT_DIRECTORY>   \\
        [--prob_cutoff  0.5]             \\
        [--threads      4]               \\
        [--max_bams     N]               \\
        [--force]

Sorties :
    <outdir>/per_sample/          – un TSV par BAM (cache intermédiaire)
    <outdir>/all_samples_counts.tsv  – fichier consolidé (tous échantillons)

Dépendances :
    pip install pysam pandas numpy
"""

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# HP tag → label
_HAP_LABELS = {0: "unphased", 1: "hap1", 2: "hap2"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement des îlots CpG
# ─────────────────────────────────────────────────────────────────────────────

def load_cpg_islands(tsv_path: str) -> pd.DataFrame:
    """
    Charge le TSV d'îlots CpG (format UCSC, optionnellement annoté closest_MANE).
    Colonnes requises : chrom, chromStart, chromEnd, cpgNum.
    Retourne un DataFrame indexé par island_id = chrom:chromStart-chromEnd.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"chrom", "chromstart", "chromend", "cpgnum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CpG TSV manque les colonnes : {missing}")

    if "closest_mane" in df.columns:
        df["island_name"] = df["closest_mane"].astype(str).str.strip()
        log.info("Nom d'îlot : colonne 'closest_mane' (symbole MANE).")
    elif "name" in df.columns:
        df["island_name"] = df["name"].astype(str).str.strip()
    else:
        df["island_name"] = ""

    df["island_id"] = (
        df["chrom"].astype(str) + ":" +
        df["chromstart"].astype(str) + "-" +
        df["chromend"].astype(str)
    )
    df = df.set_index("island_id")
    log.info("Îlots CpG chargés : %d  (%s)", len(df), tsv_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extraction des comptes 5mC depuis les tags MM/ML
# ─────────────────────────────────────────────────────────────────────────────

def _meth_in_island(
    read: pysam.AlignedSegment,
    island_start: int,
    island_end: int,
    prob_cutoff_raw: int,
) -> "tuple[int, int] | None":
    """
    Retourne (n_meth, n_unmeth) pour les positions 5mC d'un read dans
    [island_start, island_end).  Retourne None si pas d'info de méthylation.

    Utilise read.modified_bases (API pysam haut niveau) et construit le
    mapping query→ref une seule fois par read.
    """
    try:
        mod_bases = read.modified_bases
    except (AttributeError, ValueError, KeyError):
        return None

    meth_by_qpos: dict[int, int] = {}
    for (base, _strand, mod), qpos_dict in mod_bases.items():
        if base == "C" and mod == "m":
            meth_by_qpos.update(qpos_dict)

    if not meth_by_qpos:
        return None

    qpos_to_rpos: dict[int, int] = dict(read.get_aligned_pairs(matches_only=True))

    n_meth = n_unmeth = 0
    for qpos, prob in meth_by_qpos.items():
        rpos = qpos_to_rpos.get(qpos)
        if rpos is None:
            continue
        if island_start <= rpos < island_end:
            if prob >= prob_cutoff_raw:
                n_meth += 1
            else:
                n_unmeth += 1

    if n_meth + n_unmeth == 0:
        return None
    return n_meth, n_unmeth


# ─────────────────────────────────────────────────────────────────────────────
# 3. Traitement d'un BAM
# ─────────────────────────────────────────────────────────────────────────────

def process_bam(
    bam_path: str,
    cpg_df: pd.DataFrame,
    outdir: str,
    prob_cutoff: float = 0.5,
    force: bool = False,
) -> str:
    """
    Parcourt un BAM et agrège les comptes de méthylation par îlot × haplotype.

    Pour chaque groupe (îlot, haplotype) :
        frac_meth = n_cpg_meth / n_cpg_total

    Retourne le chemin du TSV de sortie.
    """
    prob_cutoff_raw = int(prob_cutoff * 255)
    sample = Path(bam_path).stem
    out_path = os.path.join(outdir, f"{sample}.counts.tsv")

    if not force and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        log.info("[%s] Ignoré – fichier déjà existant : %s", sample, out_path)
        return out_path

    log.info("[%s] Démarrage …", sample)
    t0 = time.time()

    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
    except Exception as exc:
        log.error("[%s] Impossible d'ouvrir le BAM : %s", sample, exc)
        return out_path

    n_total = len(cpg_df)
    log_every = max(1, n_total // 10)
    rows = []

    for n_done, (island_id, island) in enumerate(cpg_df.iterrows(), start=1):
        if n_done % log_every == 0 or n_done == n_total:
            log.info("[%s] Îlots parcourus : %d / %d (%.0f%%)",
                     sample, n_done, n_total, 100 * n_done / n_total)

        chrom       = island["chrom"]
        start       = int(island["chromstart"])
        end         = int(island["chromend"])
        island_cpg  = int(island["cpgnum"])
        island_name = str(island["island_name"])

        try:
            reads = bam.fetch(chrom, start, end)
        except (ValueError, KeyError):
            continue

        # acc[hp_code] = [n_meth, n_unmeth, n_reads]
        acc: dict[int, list[int]] = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}

        for read in reads:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.reference_start > start or read.reference_end < end:
                continue

            result = _meth_in_island(read, start, end, prob_cutoff_raw)
            if result is None:
                continue

            nm, nu = result
            hp = int(read.get_tag("HP")) if read.has_tag("HP") else 0
            if hp not in acc:
                hp = 0
            acc[hp][0] += nm
            acc[hp][1] += nu
            acc[hp][2] += 1

        base = dict(
            sample        = sample,
            island_id     = island_id,
            island_name   = island_name,
            chrom         = chrom,
            island_start  = start,
            island_end    = end,
            island_cpgNum = island_cpg,
        )

        # Lignes par haplotype (hap1, hap2, unphased)
        for hp_code, label in _HAP_LABELS.items():
            nm, nu, nr = acc[hp_code]
            total = nm + nu
            if total == 0:
                continue
            rows.append({**base,
                "haplotype"   : label,
                "n_reads"     : nr,
                "n_cpg_total" : total,
                "n_cpg_meth"  : nm,
                "n_cpg_unmeth": nu,
                "frac_meth"   : round(nm / total, 4),
            })

        # Ligne "all" (tous haplotypes confondus)
        all_nm     = sum(v[0] for v in acc.values())
        all_nu     = sum(v[1] for v in acc.values())
        all_reads  = sum(v[2] for v in acc.values())
        all_total  = all_nm + all_nu
        if all_total > 0:
            rows.append({**base,
                "haplotype"   : "all",
                "n_reads"     : all_reads,
                "n_cpg_total" : all_total,
                "n_cpg_meth"  : all_nm,
                "n_cpg_unmeth": all_nu,
                "frac_meth"   : round(all_nm / all_total, 4),
            })

    bam.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)

    if df.empty:
        log.warning("[%s] Aucune donnée de méthylation trouvée.", sample)
    else:
        log.info("[%s] Terminé — %d entrées, %d îlots avec données / %d total  (%.1fs)",
                 sample, len(df), df["island_id"].nunique(), n_total, time.time() - t0)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Consolidation des TSV par échantillon → fichier unique
# ─────────────────────────────────────────────────────────────────────────────

def merge_counts(
    per_sample_tsvs: list[str],
    out_path: str,
) -> pd.DataFrame:
    """
    Concatène les TSV par échantillon en un seul fichier consolidé.
    Ajoute une vérification de doublons (même sample × island_id × haplotype).
    """
    dfs = []
    for p in per_sample_tsvs:
        if p and os.path.isfile(p) and os.path.getsize(p) > 0:
            dfs.append(pd.read_csv(p, sep="\t"))
        else:
            log.warning("Ignoré (vide ou absent) : %s", p)

    if not dfs:
        log.error("Aucun TSV valide à consolider.")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # Vérification des doublons
    dup = merged.duplicated(subset=["sample", "island_id", "haplotype"])
    if dup.any():
        log.warning("%d lignes dupliquées détectées (sample × island_id × haplotype) – conservées.",
                    int(dup.sum()))

    merged.to_csv(out_path, sep="\t", index=False)
    log.info(
        "Fichier consolidé → %s\n"
        "  %d lignes  |  %d échantillons  |  %d loci uniques  |  %d groupes haplotype",
        out_path,
        len(merged),
        merged["sample"].nunique(),
        merged["island_id"].nunique(),
        merged["haplotype"].nunique(),
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 5. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bam_dir",  required=True,
                        help="Répertoire contenant les fichiers BAM (.bam + .bam.bai)")
    parser.add_argument("--cpg_tsv",  required=True,
                        help="TSV d'îlots CpG (format UCSC, annoté ou non)")
    parser.add_argument("--outdir",   required=True,
                        help="Répertoire de sortie")
    parser.add_argument("--prob_cutoff", type=float, default=0.5,
                        help="Seuil de probabilité ML pour appel méthylé [0–1] (défaut : 0.5)")
    parser.add_argument("--threads",     type=int,   default=4,
                        help="Nombre de processus parallèles (défaut : 4)")
    parser.add_argument("--max_bams",    type=int,   default=None,
                        help="Limiter au N premiers BAMs (test)")
    parser.add_argument("--force", action="store_true",
                        help="Recalculer même si les fichiers de sortie existent déjà")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    per_sample_dir = os.path.join(args.outdir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("  Détection méthylation CpG – HiFi BAMs")
    log.info("=" * 60)
    log.info("  BAM dir      : %s", args.bam_dir)
    log.info("  CpG TSV      : %s", args.cpg_tsv)
    log.info("  Outdir       : %s", args.outdir)
    log.info("  Prob cutoff  : %.2f  (= %d / 255)", args.prob_cutoff, int(args.prob_cutoff * 255))
    log.info("  Threads      : %d", args.threads)
    log.info("  Max BAMs     : %s", str(args.max_bams) if args.max_bams else "tous")
    log.info("  Force        : %s", "oui" if args.force else "non")
    log.info("=" * 60)

    # ── Chargement des îlots ──────────────────────────────────────────────────
    cpg_df = load_cpg_islands(args.cpg_tsv)

    # ── Collecte des BAMs ─────────────────────────────────────────────────────
    bam_dir   = Path(args.bam_dir)
    bam_files = sorted(bam_dir.glob("*.bam"))
    if not bam_files:
        log.error("Aucun fichier BAM trouvé dans %s", bam_dir)
        sys.exit(1)
    if args.max_bams is not None:
        bam_files = bam_files[: args.max_bams]

    log.info("%d BAM(s) à traiter :", len(bam_files))
    for b in bam_files:
        log.info("  • %s", b.name)

    # Vérification / création des index BAM
    for bam in bam_files:
        idx = Path(str(bam) + ".bai")
        if not idx.exists():
            log.warning("Index absent pour %s – indexation en cours …", bam.name)
            pysam.sort("-o", str(bam), str(bam))
            pysam.index(str(bam))
            log.info("  Indexé : %s", bam.name)

    # ── Extraction parallèle ──────────────────────────────────────────────────
    worker_args = [
        (str(b), cpg_df, per_sample_dir, args.prob_cutoff, args.force)
        for b in bam_files
    ]

    n_workers = min(args.threads, len(worker_args))
    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            per_sample_tsvs = pool.starmap(process_bam, worker_args)
    else:
        per_sample_tsvs = [process_bam(*wa) for wa in worker_args]

    n_ok = sum(
        1 for p in per_sample_tsvs
        if p and os.path.isfile(p) and os.path.getsize(p) > 0
    )
    log.info("%d / %d BAM(s) ont produit des fichiers non vides.", n_ok, len(bam_files))

    # ── Consolidation ─────────────────────────────────────────────────────────
    out_merged = os.path.join(args.outdir, "all_samples_counts.tsv")
    merge_counts(per_sample_tsvs, out_merged)

    log.info("=" * 60)
    log.info("Terminé. Fichier consolidé : %s", out_merged)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
