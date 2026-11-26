# Raw Raman conversion and APOLLO-inspired GPR preprocessing

This note accompanies [`05_raw_apollo_preprocessing.ipynb`](05_raw_apollo_preprocessing.ipynb). The conversion, APOLLO spectrum transform, feature aggregation, and GPR evaluation are implemented in [`raw_preprocessing.py`](../src/glioma_raman_gp_regression/raw_preprocessing.py).

## Raw data conversion and identity audit

The raw input has 59 `.npy` images under `data/RawData/LGm-1` through `LGm-6`. Each image has shape `(x, y, 1738)` and dtype `float32`. The merged file is [`data/raw_merged.npy`](../data/raw_merged.npy), an object array ordered to match `preprocessed.npy`; each record contains the original 3D `data`, current `name`, folder-derived `class_id`, filename-derived `patient_id`, and relative `source_file`. Derived `pca` and `tsne` values are not copied because they do not apply to raw spectra.

All 59 names matched exactly, with no missing or extra samples. For every file, the patient ID parsed from `HF-<digits>` and class ID parsed from `LGm-<n>` matched the current entry. All spatial dimensions matched the corresponding processed image, while the spectral width changed from 1,738 to 1,167. The 59 images contain 300,506 pixel spectra in total, matching the number reported for APOLLO. The merge was checked by SHA-256 hashes of every array against its source file; all pixel values and dtypes are preserved. The output is 2,089,125,085 bytes, and a 59-row manifest is written to `data/raw_merged_manifest.csv`.

## APOLLO preprocessing reproduced

Lita et al. report removing the start of the spectrum through 248.651529 cm⁻¹ and the silent region from 1797.210 to 2697.804 cm⁻¹; applying airPLS baseline correction to each retained region with lambda 4 and p-order 1; setting negative residuals to zero; concatenating the two regions; and L2-normalizing each spectrum. They also describe DBSCAN-based tumor/non-tumor spot curation with ε = 0.28 and exclusion of sample HF-1887 after quality review. [Lita et al., *Neuro-Oncology* (2024)](https://doi.org/10.1093/neuonc/noae101)

The raw `.npy` files do not include a separate wavenumber calibration vector. The channel slices used here are `[104:907]` and `[1374:1738]`, which retain 1,167 channels. Applying the APOLLO airPLS and normalization steps to five evenly spaced pixels in each of the 59 images (295 spectra total) reproduced the corresponding current processed spectra to floating-point precision: maximum absolute difference was `8.1e-16`, with median cosine similarity 1.0 in every entry. This verifies the spectral transform on sampled pixels; it does not compare every raw pixel by re-running airPLS. The current `preprocessed.npy` therefore serves as the APOLLO-processed baseline.

The comparison below does not apply APOLLO's DBSCAN tumor-spot selection. The available arrays lack H&E-validated masks and acquisition annotations needed to verify that a selected spectral cluster corresponds to tumor rather than biological heterogeneity. All patients, including HF-1887, remain in the patient-level analysis.

## Preprocessing experiments

Each alternative was applied to individual spectra first, then spectra from each patient's entries were pooled into one patient mean. Five fixed representations were compared:

1. **APOLLO current:** the existing processed spectra, which reproduce the paper's baseline correction, clipping, crop, and L2 normalization on the sampled check.
2. **Raw crop + clip + L2, no airPLS:** use the raw spectra, retain the APOLLO channel ranges, clip negative values, and L2-normalize. “No airPLS” refers only to omitting the software airPLS step; the raw data may already reflect instrument-level baseline correction.
3. **APOLLO + pixel SNV:** after APOLLO processing, center and standardize each individual spectrum, then L2-normalize.
4. **APOLLO + Savitzky–Golay smoothing:** apply an 11-channel, third-order Savitzky–Golay smoother, then L2-normalize.
5. **APOLLO + Savitzky–Golay first derivative:** calculate the first derivative with the same window/order, then L2-normalize.

The same patient-level PCA 8 + ARD RBF GPR was fitted for all five representations. PCA and GPR were refitted within each training fold. Five-fold stratified CV was repeated with seeds 42, 101, and 202; each patient's held-out predictions were pooled within each repeat before scoring. Seed 42 uses the PCA solver and GP optimizer seed from notebook 03, reproducing its APOLLO baseline result exactly. Mean and SD below are across the three split seeds, not confidence intervals.

## GPR results

| Preprocessing | RMSE | MAE | Gaussian NLPD | 90% coverage | 90% width | Accuracy | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| APOLLO current | 0.423 (0.021) | 0.309 (0.010) | 0.679 (0.151) | 0.797 (0.013) | 1.126 (0.019) | 0.739 (0.000) | 0.727 (0.002) | 0.813 (0.041) |
| Raw crop + clip + L2, no airPLS | 0.439 (0.045) | 0.301 (0.035) | 5.036 (3.563) | 0.739 (0.075) | 0.826 (0.094) | 0.739 (0.058) | 0.731 (0.061) | 0.813 (0.042) |
| APOLLO + pixel SNV | 0.441 (0.046) | 0.328 (0.023) | 5.858 (4.458) | 0.790 (0.013) | 1.061 (0.061) | 0.732 (0.045) | 0.722 (0.040) | 0.805 (0.037) |
| APOLLO + Savitzky–Golay smooth | 0.430 (0.028) | 0.330 (0.028) | 1.004 (0.566) | 0.812 (0.050) | 1.150 (0.111) | 0.703 (0.033) | 0.687 (0.038) | 0.797 (0.058) |
| APOLLO + Savitzky–Golay first derivative | **0.410 (0.027)** | 0.308 (0.013) | **0.613 (0.136)** | 0.804 (0.038) | 1.194 (0.019) | 0.739 (0.065) | **0.728 (0.066)** | **0.829 (0.027)** |

The APOLLO baseline seed-42 result reproduces notebook 03: accuracy 0.739, balanced accuracy 0.730, and AUC 0.840. Across repeats, the Savitzky–Golay derivative had the best mean RMSE, NLPD, and AUC. Its mean accuracy was unchanged from APOLLO, and accuracy varied more across seeds (SD 0.065). This is a modest ranking/forecast improvement worth retesting, not a reliable accuracy gain.

The raw crop without airPLS tied APOLLO on mean accuracy and AUC, with slightly higher balanced accuracy, but it produced narrow intervals, lower coverage, and poor/unstable Gaussian NLPD. Pixel SNV also had unstable and poor NLPD despite an AUC close to APOLLO. Because regression-derived threshold-exceedance values are not calibrated Bernoulli probabilities, the predictive density and interval diagnostics matter; a preprocessing choice should not be selected from AUC alone.

## Conclusions and limitations

- The current `preprocessed.npy` is already APOLLO-style preprocessing: the sampled raw-to-processed outputs match at floating-point precision. Re-running airPLS on every raw pixel is unnecessary for this comparison.
- Savitzky–Golay first derivatives are the strongest candidate in these tests, but their mean AUC gain over APOLLO is small (0.829 vs 0.813) and mean accuracy did not increase. Test this choice inside nested patient-level validation before making a performance claim.
- Omitting airPLS and applying SNV did not yield stable improvements in the predictive distribution. Large NLPD and narrow intervals indicate overconfident fits on some splits.
- These models use only 46 independent patients. The three repeated split seeds measure split sensitivity, not external generalization. All comparisons are exploratory and share the same cohort.
- APOLLO's DBSCAN tumor/non-tumor filtering and HF-1887 quality exclusion were not reproduced here. A separate artifact-curation study would need a validated way to map clusters to histology and must retain patient-level separation in CV.

## Reference

Adrian Lita et al., “Raman-based machine-learning platform reveals unique metabolic differences between IDHmut and IDHwt glioma,” *Neuro-Oncology* 26(11), 1994–2009 (2024). [Article](https://doi.org/10.1093/neuonc/noae101)
