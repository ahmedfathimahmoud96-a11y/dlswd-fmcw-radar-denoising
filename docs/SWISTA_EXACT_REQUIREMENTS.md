# SWISTA-Net Baseline Status

The current repository now contains `matlab/run_swista_nets_baseline.m`, a measured SWISTA-Net baseline for the FMCW matrix denoising task.

## Implemented Baseline

- Seven unrolled wavelet-shrinkage iterations.
- Three-level subband-adaptive Haar decomposition.
- Trainable per-stage/subband step sizes and thresholds.
- Trainable FISTA-style momentum.
- Learned three-layer convolutional post-processing path.
- Identity forward operator, because this radar experiment provides paired noisy/clean FMCW matrices rather than the medical inverse-reconstruction forward model.
- Five source-frame-disjoint folds, evaluated on all 17,940 held-out records across the folds.

The measured five-fold run reports:

- Correlation: `0.6669`
- Output SNR: `4.6117 dB`
- MSE: `441.24`

Primary outputs:

- `matlab/out/swista_nets/swista_nets_summary.csv`
- `matlab/out/swista_nets/swista_nets_per_record.csv`
- `matlab/out/swista_nets/swista_nets_by_snr.csv`

## Paper Identity

- Binchun Lu, Lidan Fu, Yixuan Pan, and Yonggui Dong, "SWISTA-Nets: Subband-adaptive wavelet iterative shrinkage thresholding networks for image reconstruction," Computerized Medical Imaging and Graphics, vol. 113, article 102345, 2024.
- DOI: `10.1016/j.compmedimag.2024.102345`

## Guard Rail

Do not describe the older `matlab/run_swista_style_baseline.m` script as SWISTA-Nets. For public reporting, use the new `run_swista_nets_baseline.m` result as a SWISTA-Net adaptation retrained on the FMCW matrices. The SWISTA-Net_a learned-DWT variant and the original medical-image forward operator are not claimed in this package.
