# GPR uncertainty, robustness, and patient-entry structure

This note accompanies [`04_gpr_uncertainty_tests.ipynb`](04_gpr_uncertainty_tests.ipynb). The implementation is in [`gpr_followup.py`](../src/glioma_raman_gp_regression/gpr_followup.py). It extends the patient-level regression work in notebook 03 while keeping the Gaussian process regressor as the model under study.

## Data and evaluation

The input contains 59 sample entries from 46 patients (21 IDH-mutant, 25 IDH-wildtype), with 1,167 spectral channels. The target is 0 for mutant and 1 for wildtype. Every split is stratified and patient-level. Each experiment uses five folds with seeds 42, 101, and 202; PCA and feature scaling are fitted on training patients only. The three repeat scores are summarized as mean and standard deviation. They describe sensitivity to the split and are not independent confidence intervals.

Regression forecast measures are primary: RMSE, MAE, predictive negative log density (NLPD), and 90% interval coverage and width. AUC, balanced accuracy, and accuracy at a 0.5 regression cutoff are secondary views of separation. GPR threshold-exceedance values are not calibrated binary class probabilities.

## Tests and results

### Kernel and split stability

Compared PCA 6, 8, 12, and 20 with ARD RBF kernels and PCA 8 and 12 with isotropic RBF kernels. GPR kernel hyperparameters were optimized within each training fold with two optimizer restarts. The table reports mean (SD) across the three split seeds; the bound column is the fraction of the 15 fits where at least one kernel parameter was at a configured bound.

| Model | RMSE | MAE | Gaussian NLPD | 90% coverage | ROC AUC | Fits at a bound |
|---|---:|---:|---:|---:|---:|---:|
| PCA 6 + ARD RBF | 0.441 (0.019) | 0.337 (0.014) | 1.965 (1.710) | 0.776 | 0.788 (0.024) | 93% |
| PCA 8 + ARD RBF | 0.451 (0.022) | 0.359 (0.017) | 2.086 (1.890) | 0.739 | 0.738 (0.021) | 100% |
| PCA 12 + ARD RBF | 0.442 (0.016) | 0.357 (0.009) | 1.909 (1.371) | 0.748 | 0.764 (0.032) | 100% |
| PCA 20 + ARD RBF | 0.449 (0.065) | 0.344 (0.050) | 4.609 (3.426) | 0.712 | 0.785 (0.072) | 100% |
| PCA 8 + isotropic RBF | 0.465 (0.005) | 0.390 (0.005) | 0.700 (0.032) | 0.826 | 0.693 (0.032) | 20% |
| PCA 12 + isotropic RBF | 0.447 (0.011) | 0.378 (0.006) | 0.723 (0.139) | 0.870 | 0.753 (0.043) | 0% |

ARD fits frequently reached kernel bounds, including every PCA 20 fit. PCA 20 had the highest mean AUC among the ARD options, but also the largest split-to-split variation in RMSE and AUC and very variable Gaussian NLPD. The isotropic fits were less prone to bounds and had better coverage/NLPD, at the cost of weaker ranking for PCA 8. This supports treating kernel complexity and predictive uncertainty as part of the result, rather than choosing a component count from accuracy alone.

### Robust regression with Student-t IRLS

The Student-t experiment uses a fixed `df=4` iterative reweighted least squares approximation. It first fits Gaussian GPR on PCA 8, freezes the learned signal kernel and noise scale, then iteratively increases effective observation noise for patients with large standardized residuals. This is **not exact Student-t likelihood inference**; it is a robust GPR sensitivity analysis that keeps observations in the fit instead of deleting them.

| Model | RMSE | MAE | Predictive NLPD | 90% coverage | 90% width | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian GPR, PCA 8 ARD | 0.447 (0.016) | 0.358 (0.015) | 1.591 (1.065), Gaussian | 0.732 (0.050) | 1.090 | 0.667 (0.033) | 0.741 (0.018) |
| Student-t IRLS approximation, df 4 | 0.447 (0.018) | 0.354 (0.019) | 0.750 (0.150), Student-t | 0.841 (0.030) | 1.416 | 0.685 (0.044) | 0.748 (0.026) |

The point forecasts and AUC changed little. The robust approximation produced wider intervals and higher observed coverage. The likelihood scores use their stated predictive distributions, so they should be read together with the coverage and width rather than as evidence of a large separation gain. This test does not establish that any patient is erroneous or should be excluded.

### Heteroscedastic noise sensitivity

The preprocessed file has no technical replicate noise estimates. As a sensitivity test, the notebook calculates each patient's within-patient variance in the fold-fitted PCA space across its image pixels. It converts this spread to a bounded relative proxy, shrinks it toward the training median, and uses it to distribute a training-fold noise level across patients. This proxy mixes spatial and biological heterogeneity with measurement noise; it is **not** a direct estimate of output-label or instrument noise.

| Model | RMSE | MAE | Gaussian NLPD | 90% coverage | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Gaussian GPR, PCA 8 ARD | 0.447 (0.016) | 0.358 (0.015) | 1.591 (1.065) | 0.732 (0.050) | 0.741 (0.018) |
| Spread-proxy heteroscedastic GPR | 0.448 (0.017) | 0.358 (0.016) | 1.619 (1.082) | 0.740 (0.054) | 0.734 (0.016) |

The proxy did not improve the regression or separation metrics. The near tie is more informative as a limitation: a better noise model needs technical replicate or QC information, not just within-patient spectral spread.

### Patient-entry bag kernel

Each patient remains one training observation with one binary target. The bag kernel compares two patients by averaging an RBF kernel across every pair of their entry-mean spectra. A matched patient-mean GP uses the same PCA, equal-patient feature scaling, covariance family, and fold-wise marginal-likelihood fitting, but represents each patient by one mean vector. Thus, repeated entries contribute to the patient's feature distribution without adding extra labeled rows.

| Representation | RMSE | MAE | Gaussian NLPD | 90% coverage | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| One patient mean | 0.413 (0.024) | 0.338 (0.011) | 0.583 (0.152) | 0.878 (0.044) | 0.642 (0.060) | 0.845 (0.037) |
| Patient-entry bag kernel | 0.401 (0.020) | 0.324 (0.009) | 0.541 (0.114) | 0.893 (0.041) | 0.670 (0.049) | 0.868 (0.028) |

The bag kernel had lower RMSE and higher AUC in each of the three split seeds. Mean AUC was 0.900, 0.848, and 0.856 by seed, versus 0.888, 0.820, and 0.828 for the patient-mean kernel. This is the clearest positive result in this follow-up, but it remains exploratory: only nine patients have multiple entries, and the model representation has not been selected inside a nested outer evaluation or checked on an external cohort.

## Conclusions and next step

- The PCA 20 ARD result remains fragile: optimization hit a bound in every fit, with substantial split variation and unstable predictive density.
- The Student-t approximation modestly improved coverage and separation metrics while widening intervals. It is a useful robustness sensitivity analysis, not an exact Student-t GP posterior.
- The input-spread noise proxy did not improve performance. Technical noise estimates would be needed for a stronger heteroscedastic test.
- The patient-entry bag kernel is promising because it improved regression error and ranking while preserving one vote per patient. The next defensible experiment is to compare bag and mean representations inside the inner folds of repeated nested patient CV, with the model family fixed before the outer evaluation. External validation is still needed for a generalization claim.

All values are exploratory on 46 patients. Three repeated split seeds are not independent validation cohorts, and the regression-derived `P(output > 0.5)` values remain different from calibrated Bernoulli class probabilities.

## References

- P. Jylänki, J. Vanhatalo, and A. Vehtari, “Robust Gaussian Process Regression with a Student-t Likelihood,” *Journal of Machine Learning Research*, 2011. [Paper](https://www.jmlr.org/beta/papers/v12/jylanki11a.html)
- D. A. Goldberg, C. K. I. Williams, and C. M. Bishop, “Regression with Input-Dependent Noise: A Gaussian Process Treatment,” NeurIPS, 1997. [Paper](https://papers.nips.cc/paper_files/paper/1997/hash/afe434653a898da20044041262b3ac74-Abstract.html)
