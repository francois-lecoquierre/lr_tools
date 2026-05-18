# lr_tools
Tools for human long-read genome sequencing experiments

# xci_from_bam

Haplotype-specific CpG island methylation analysis on chrX for X-chromosome inactivation (XCI) bias detection from haplotagged PacBio HiFi BAM files.

## Overview

For each non-pseudoautosomal CpG island on chrX, the script:

1. Selects reads that fully span the island.
2. Classifies each read as **methylated**, **unmethylated**, or **partial** based on the fraction of methylated CpGs (MM/ML tags).
3. Aggregates results per island per haplotype (WhatsHap or HiPhase `HP` tag).
4. Detects hemi-methylated islands — the hallmark of XCI — as islands where the two haplotypes show opposite methylation patterns.

## Requirements

- Python ≥ 3.10
- [pysam](https://github.com/pysam-developers/pysam)
- pandas
- numpy
- plotly
- scipy

Install with:

```bash
pip install pysam pandas numpy plotly scipy
