# Data Requirements

This repository does not include raw radar data. Start by downloading the public Gao raw ADC FMCW radar dataset, then place the files in the local layout expected by the regeneration scripts.

## 1. Download the Public Dataset

Use either source:

- Gao raw ADC radar dataset GitHub: https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection
- IEEE DataPort DOI: https://dx.doi.org/10.21227/xm40-jx59

The relevant sequence for this package is:

```text
2019_04_09_pms1000
```

The raw radar frames are stored as `.mat` files under `radar_raw_frame`, and text labels are stored under `text_labels`. The expected raw ADC variable is `adcData` with size `128 x 255 x 4 x 2`.

## 2. Recommended Local Layout

Use this local root unless you intentionally edit the MATLAB paths:

```text
F:\Master\1-MATLAB\PED_Wavelet_Denoising
```

Recommended raw public-data staging:

```text
public_release\tmp\public_pms\000030.mat
public_release\tmp\public_pms\000031.mat
...
public_release\tmp\public_pms\000059.mat

public_release\matlab\out\cached_grouped_revision\public_pms1000_text_labels\0000000030.csv
public_release\matlab\out\cached_grouped_revision\public_pms1000_text_labels\0000000031.csv
...
public_release\matlab\out\cached_grouped_revision\public_pms1000_text_labels\0000000059.csv
```

The `tmp/` and `matlab/out/` folders are intentionally ignored by Git.

## 3. Generated Clean/Noisy Matrix Layout

The DLSWD grouped workflow expects or regenerates clean and noisy pedestrian CSV matrices using this layout:

```text
F:\Master\1-MATLAB\PED_Wavelet_Denoising\Dataset\Dataset_original\ped\CSV\ped30.csv
F:\Master\1-MATLAB\PED_Wavelet_Denoising\Dataset\Dataset_noisy\train\ped\CSV\SNR-20\ped30_-20_01.csv
F:\Master\1-MATLAB\PED_Wavelet_Denoising\Dataset_Test\Dataset_noisy\test\ped\CSV\SNR-20\ped30_-20_01.csv
```

The included `results/cached/manifest.csv` documents the source frame, SNR level, replicate index, clean path, noisy path, fold assignment, and selected denoising configuration for each cached row.

## 4. Needed Before Claiming Target-Level Metrics

The Gao label files support object class and position parsing, but the current package does not treat them as verified radial-velocity ground truth. Do not claim empirical range-velocity error unless you add and document a verified conversion from labels to radial-velocity truth.

Additional experiments are also required before claiming coverage of non-AWGN clutter, multipath, mutual interference, multi-target rare events, or safety-critical long-tail scenarios.