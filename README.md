# DLSWD FMCW Radar Denoising Reproducibility Package

This repository is a standalone code and results package for reproducing the DLSWD FMCW radar denoising workflow. It does not include article PDF/source files or private radar data. The workflow starts from the public Gao raw-ADC FMCW radar dataset, builds clean/noisy FMCW matrix records, runs wavelet-configuration selection and baseline comparisons, and regenerates the cached tables and figures included here.

## What Is Included

- MATLAB scripts for DLSWD wavelet-configuration search, CBC filtering, grouped evaluation, SNR/correlation sweeps, Golilarz-style threshold control, conventional filters, SWISTA-Net baseline, projected detection metrics, and RD-CFAR audit.
- Cached CSV outputs under `results/` so the reported numbers can be inspected without rerunning every expensive experiment.
- Figure outputs under `figures/`, `figures_ieee/`, and `Ped/`.
- KR260 CNN-only timing/profile helper under `matlab/board/`.
- Documentation for dataset placement and full regeneration.

## What Is Not Included

- No article PDF or source files.
- No raw Gao dataset files or local generated noisy CSV datasets, because those files are large and must be obtained from the public dataset source.
- No unsupported target-level range-velocity error claims. The Gao text labels used here support annotation inventory and range/angle checks, but not independently verified radial-velocity ground truth.

## Dataset Source

Download the public dataset from the Gao raw ADC radar dataset project:

- GitHub: https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection
- IEEE DataPort DOI: https://dx.doi.org/10.21227/xm40-jx59

The expected raw frame shape is `128 x 255 x 4 x 2` for samples, chirps, receivers, and transmitters. The DLSWD study uses the pedestrian sequence `2019_04_09_pms1000`, frames `000030.mat` through `000059.mat`, and one virtual antenna channel for the clean source matrices.

## Quick Start

1. Install MATLAB R2021b or later with Wavelet Toolbox, Deep Learning Toolbox, and Signal Processing Toolbox.
2. Download the Gao dataset and place the pedestrian raw frames and labels using the layout in `docs/DATA_REQUIREMENTS.md`.
3. Open MATLAB from the repository root.
4. Run:

```matlab
cd matlab
run_cached_grouped_revision
make_sota_by_snr_summary
make_sota_snr_sweep_figure
```

For the learned baseline and radar-product audit, run:

```matlab
run_swista_nets_baseline('Fold',"all",'Epochs',2)
run_golilarz_style_baseline
evaluate_cached_radar_operating_metrics
evaluate_labelled_pms1000_operating_metrics
export_gao_pms_rd_examples
```

The scripts write regenerated outputs under `matlab/out/`. Cached public outputs are already provided under `results/` for inspection.

## Main Outputs

- `results/cached/sota_comparison_by_snr.csv`
- `results/cached/sota_comparison_by_snr_band.csv`
- `results/cached/cnn_vs_lookup_paired_tests.csv`
- `results/cached/cbc_sensitivity.csv`
- `results/swista_nets/swista_nets_summary.csv`
- `results/cached/radar_operating_summary.csv`
- `results/gao_pms/pms1000_label_inventory.csv`

## Full Workflow

See `docs/REPRODUCTION_WORKFLOW.md` for the complete path from downloading the Gao dataset to regenerating the cached comparison tables and figures.

## Repository Notes

Keep all raw `.mat` files, generated noisy CSV matrices, trained network checkpoints, and local temporary folders outside Git. The `.gitignore` file is configured for this.