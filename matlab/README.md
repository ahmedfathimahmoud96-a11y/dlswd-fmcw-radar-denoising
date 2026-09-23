# DLSWD MATLAB Reproduction Pipeline

This folder contains the MATLAB scripts used to regenerate the public DLSWD FMCW radar denoising results.

## Requirements

MATLAB R2021b or later with Wavelet Toolbox, Deep Learning Toolbox, and Signal Processing Toolbox. Parallel Computing Toolbox is optional but useful for the wavelet-cache stages.

## Data Setup

First download the Gao raw ADC dataset and stage the pedestrian sequence as described in `../docs/DATA_REQUIREMENTS.md`. The grouped DLSWD scripts expect clean/noisy matrix CSV files under the documented `F:/Master/1-MATLAB/PED_Wavelet_Denoising` layout unless you edit the path variables.

## Fast Path Using Supplied Cached Results

When the precomputed CSV matrices are available, use:

```matlab
cd matlab
run_cached_grouped_revision
```

This script reuses the cached three-class clean-reference labels, merges the legacy train/test folders as 23 noisy realizations per source/SNR pair, enforces five source-frame-disjoint folds, runs five CNN seeds per fold, and writes outputs under `matlab/out/cached_grouped_revision/`.

## Comparison and Figure Scripts

```matlab
make_sota_by_snr_summary
make_sota_snr_sweep_figure
run_conventional_filter_comparison
run_golilarz_style_baseline
run_swista_nets_baseline('Fold',"all",'Epochs',2)
evaluate_cached_radar_operating_metrics
evaluate_labelled_pms1000_operating_metrics
export_gao_pms_rd_examples
```

## Included Evidence

- DLSWD-CNN versus DLSWD-LUT paired significance-test CSVs.
- Conventional-filter and fixed-wavelet comparison summaries.
- Golilarz-style same-data baseline summaries.
- SWISTA-Net cached per-record, per-source, and per-SNR outputs.
- Projected radar operating-metric and RD-CFAR audit summaries.

## Limits

The supplied folders do not include independently verified physical radial-velocity target annotations. The current workflow therefore does not claim range-velocity accuracy; it reports reconstruction metrics, projected radar operating metrics, RD-CFAR audit summaries, and KR260 CNN-only deployment evidence.

## Figure Style

`ieee/ieeeStyle.m` defines the common figure style: Times New Roman, IEEE column widths, high-resolution PNG plus vector PDF export, and a greyscale-separable color palette.