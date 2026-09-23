# Full Reproduction Workflow

This document describes the standalone path from the public Gao raw ADC FMCW radar dataset to the cached DLSWD comparison outputs.

## Step 1: Get the Gao Raw ADC Dataset

Download the dataset from one of the official public locations:

- https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection
- https://dx.doi.org/10.21227/xm40-jx59

Use the pedestrian sequence `2019_04_09_pms1000`. For the DLSWD source-frame benchmark, use frames `000030.mat` through `000059.mat`.

## Step 2: Stage Raw Frames and Labels

Place raw frames here:

```text
public_release/tmp/public_pms/000030.mat
...
public_release/tmp/public_pms/000059.mat
```

Place corresponding text labels here:

```text
public_release/matlab/out/cached_grouped_revision/public_pms1000_text_labels/0000000030.csv
...
public_release/matlab/out/cached_grouped_revision/public_pms1000_text_labels/0000000059.csv
```

These folders are ignored by Git because they hold downloaded or generated data.

## Step 3: Build Clean Source Matrices

Convert the selected raw ADC frames into clean pedestrian CSV matrices under:

```text
F:/Master/1-MATLAB/PED_Wavelet_Denoising/Dataset/Dataset_original/ped/CSV/
```

Each clean matrix should preserve the slow-time by fast-time radar representation used by the DLSWD scripts. The included cached manifest records the expected source-frame identifiers.

## Step 4: Generate Noisy AWGN Matrix Sets

Generate noisy matrix realizations across input SNR levels `-20:1:5` dB, with 23 noisy realizations per source-frame/SNR pair. The grouped workflow expects the legacy train/test folders:

```text
F:/Master/1-MATLAB/PED_Wavelet_Denoising/Dataset/Dataset_noisy/train/ped/CSV/
F:/Master/1-MATLAB/PED_Wavelet_Denoising/Dataset_Test/Dataset_noisy/test/ped/CSV/
```

## Step 5: Run DLSWD Grouped Evaluation

From MATLAB:

```matlab
cd matlab
run_cached_grouped_revision
```

This evaluates the cached three-class configuration labels with source-frame-disjoint folds and writes grouped DLSWD-CNN and DLSWD-LUT outputs under `matlab/out/cached_grouped_revision/`.

## Step 6: Run Baselines

```matlab
run_conventional_filter_comparison
run_golilarz_style_baseline
run_swista_nets_baseline('Fold',"all",'Epochs',2)
```

The SWISTA-Net run can take longer than the cached-table path. Cached outputs are included under `results/swista_nets/`.

## Step 7: Build SNR-Sweep Tables and Figures

```matlab
make_sota_by_snr_summary
make_sota_snr_sweep_figure
make_final_pd_singlepfa_figure
```

Expected output tables include:

```text
results/cached/sota_comparison_by_snr.csv
results/cached/sota_comparison_by_snr_band.csv
```

Expected figures include:

```text
figures_ieee/F_sota_snr_improvement.pdf
figures_ieee/F_sota_correlation_preservation.pdf
Ped/F_labelled_Pd_vs_SNR_singlePfa.png
```

## Step 8: Run Radar-Product Audit

```matlab
evaluate_cached_radar_operating_metrics
evaluate_labelled_pms1000_operating_metrics
export_gao_pms_rd_examples
```

These scripts generate projected operating metrics, RD-CFAR audit summaries, label inventories, and representative range-Doppler maps. They do not claim radial-velocity error because the required velocity ground truth is not independently verified in this package.

## Step 9: Inspect Cached Results

The public cached outputs are copied to:

```text
results/cached/
results/swista_nets/
results/gao_pms/
```

Use these files for quick verification before rerunning expensive training or full wavelet sweeps.