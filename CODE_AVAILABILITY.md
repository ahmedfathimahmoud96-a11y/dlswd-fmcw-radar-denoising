# Code and Reproducibility Package

This repository provides the standalone public code package for the DLSWD FMCW radar denoising workflow. It is intended to be usable as a code-first repository, without attaching article source files or PDFs.

## Public Repository URL


## Included Code

- `matlab/run_cached_grouped_revision.m`: grouped DLSWD-CNN and DLSWD-LUT evaluation.
- `matlab/run_conventional_filter_comparison.m`: conventional non-adaptive filter controls.
- `matlab/run_golilarz_style_baseline.m`: same-data adaptive-threshold control inspired by Golilarz et al.
- `matlab/run_swista_nets_baseline.m`: retrained SWISTA-Net learned wavelet-shrinkage baseline.
- `matlab/make_sota_by_snr_summary.m`: combines SNR-sweep comparison rows.
- `matlab/make_sota_snr_sweep_figure.m`: regenerates SNR and correlation comparison figures.
- `matlab/evaluate_cached_radar_operating_metrics.m`: projected detection and RD-CFAR audit from cached records.
- `matlab/evaluate_labelled_pms1000_operating_metrics.m`: label-backed Gao PMS audit support.
- `matlab/export_gao_pms_rd_examples.m`: representative Gao range-Doppler map export.
- Helper functions under `matlab/lib/` and figure-style helpers under `matlab/ieee/`.

## Included Cached Results

- `results/cached/sota_comparison_by_snr.csv`
- `results/cached/sota_comparison_by_snr_band.csv`
- `results/cached/cnn_vs_lookup_paired_tests.csv`
- `results/cached/cnn_vs_lookup_paired_ci.csv`
- `results/cached/golilarz_style_summary.csv`
- `results/cached/conventional_filter_summary.csv`
- `results/cached/radar_operating_summary.csv`
- `results/cached/radar_operating_by_source.csv`
- `results/cached/radar_operating_policy_agreement.csv`
- `results/cached/labelled_pms1000_operating_summary.csv`
- `results/cached/labelled_pms1000_operating_by_source.csv`
- `results/cached/cnn_seed_metrics.csv`
- `results/cached/cbc_sensitivity.csv`
- `results/cached/configuration_library.csv`
- `results/cached/configuration_selection_by_fold.csv`
- SWISTA-Net cached outputs under `results/swista_nets/`.

## External Data

Raw radar data is not committed. Download it from the public Gao raw ADC radar dataset:

- https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection
- https://dx.doi.org/10.21227/xm40-jx59

The local regeneration root used by the scripts can be set to:

```text
F:\Master\1-MATLAB\PED_Wavelet_Denoising
```

The cached manifest has been remapped to the `F:` layout. If your dataset is elsewhere, either mirror the documented folder structure or update the path variables in the MATLAB scripts before rerunning the full workflow.

## Scope Limits

This package supports reconstruction metrics, SNR/correlation sweeps, CBC sensitivity, same-data baselines, projected operating metrics, RD-CFAR audit summaries, and KR260 CNN-only profiling. It does not claim target-annotation-backed range-velocity error or independently verified radial-velocity ground truth.