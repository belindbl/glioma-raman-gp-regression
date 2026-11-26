# Patient-level Gaussian process regression for IDH status

This note accompanies [`03_gaussian_process_regression_test.ipynb`](03_gaussian_process_regression_test.ipynb). It uses the class mapping and patient-level five-fold split established in [`01_data_exploration.ipynb`](01_data_exploration.ipynb) to compare Gaussian process regression (GPR) choices, test candidate outlier handling, and evaluate the six follow-up approaches below for IDH-mutant versus IDH-wildtype separation.

## Data and validation design

The preprocessed file contains 59 sample entries from 46 patients, with 1,167 intensity channels per spectrum. Classes 1-3 are mapped to IDH-mutant and classes 4-6 to IDH-wildtype. This yields 21 mutant and 25 wildtype patients.

Each patient is represented by one spectrum: all spectra from all of that patient's entries are averaged channel by channel. The patient, rather than an individual spectrum, is the observation and split unit. This prevents spectra from the same patient appearing in both training and validation data and gives each patient equal weight. It does discard within-patient and within-entry variation.

The notebook reconstructs the original `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` assignment from the patient table sorted by patient ID, as in notebook 01. PCA, optional channel scaling, outlier detection, and GP fitting are repeated using only each fold's training patients. Every validation patient remains in the scored predictions, including patients flagged as outliers in other folds.

## Regression setup and comparisons

The binary target is encoded as 0 for mutant and 1 for wildtype. GPR is fit to this numeric target. In addition to the raw regression mean, the notebook reports a regression-derived threshold-exceedance probability: if the predictive output is approximately normal with mean `mu` and standard deviation `sigma`, then `P(IDH-wildtype) = P(output > 0.5) = Phi((mu - 0.5) / sigma)`, and `P(IDH-mutant) = 1 - P(IDH-wildtype)`. These probabilities sum to one and show how much predictive mass falls on either side of the target boundary. They are not calibrated posterior probabilities for the binary class, because GPR uses a Gaussian regression likelihood rather than a Bernoulli likelihood. The accuracy metrics still use a 0.5 cutoff on the regression mean; ROC AUC uses the continuous held-out means. Metrics are pooled from one out-of-fold prediction per patient.

The notebook compares 2, 5, 6, 8, 10, 12, 15, 20, and 25 PCA components; ARD and isotropic RBF kernels; an ARD Matérn kernel with smoothness `nu=1.5`; an isotropic Rational Quadratic kernel; and an alternative that standardizes each spectral channel before PCA. All transformations are fit on the training fold. The kernels include an optimized signal variance and a white-noise term. The GP uses `normalize_y=True`, `alpha=1e-6`, and one optimizer restart. The same five folds are used for every candidate, so these results compare methods on this cohort but do not constitute an independent final estimate after model selection.

### Five-fold out-of-fold results

| Method | Accuracy | Balanced accuracy | ROC AUC | Confusion matrix (mutant row, wildtype row) |
|---|---:|---:|---:|---:|
| PCA 2 + RBF ARD | 0.522 | 0.488 | 0.434 | `[[2, 19], [3, 22]]` |
| PCA 5 + RBF ARD | 0.630 | 0.614 | 0.657 | `[[9, 12], [5, 20]]` |
| PCA 6 + RBF ARD | 0.739 | 0.730 | **0.867** | `[[13, 8], [4, 21]]` |
| PCA 8 + RBF ARD | 0.739 | 0.730 | 0.840 | `[[13, 8], [4, 21]]` |
| PCA 10 + RBF ARD | 0.739 | 0.730 | 0.846 | `[[13, 8], [4, 21]]` |
| PCA 12 + RBF ARD | 0.739 | 0.730 | 0.851 | `[[13, 8], [4, 21]]` |
| PCA 15 + RBF ARD | 0.739 | 0.733 | 0.850 | `[[14, 7], [5, 20]]` |
| PCA 20 + RBF ARD | **0.804** | **0.797** | 0.821 | `[[15, 6], [3, 22]]` |
| PCA 25 + RBF ARD | 0.696 | 0.693 | 0.766 | `[[14, 7], [7, 18]]` |
| PCA 10 + isotropic RBF | 0.609 | 0.594 | 0.712 | `[[9, 12], [6, 19]]` |
| PCA 10 + ARD Matérn (`nu=1.5`) | 0.717 | 0.702 | 0.792 | `[[11, 10], [3, 22]]` |
| PCA 10 + isotropic Rational Quadratic | 0.587 | 0.570 | 0.701 | `[[8, 13], [6, 19]]` |
| Channel-standardized PCA 10 + RBF ARD | 0.543 | 0.500 | 0.550 | `[[0, 21], [0, 25]]` |
| PCA 10 + RBF ARD, training outliers removed | 0.652 | 0.627 | 0.698 | `[[7, 14], [2, 23]]` |
| PCA 10 + RBF ARD, patient 1887 removed from training | 0.739 | 0.722 | 0.771 | `[[11, 10], [2, 23]]` |

PCA 20 has the highest accuracy in this comparison (37/46 patients), while PCA 6 has the highest ROC AUC. PCA 8 is not a poor test: it ties PCA 10 on accuracy and balanced accuracy, though its AUC is slightly lower. PCA 6, 8, 10, and 12 give the same aggregate confusion matrix and accuracy here, while their ranking metrics vary. Accuracy is not monotonic with component count: PCA 25 drops to 0.696. At least one optimized kernel parameter reached a bound in all five PCA 20 fits, which is a warning that its apparent gain may be fragile. Accuracy also varies across folds: it ranges from 0.70 to 1.00 for PCA 20 and from 0.67 to 0.89 for PCA 10.

Channel-wise standardization performed poorly in this experiment: all 46 held-out predictions were wildtype. Equalizing the variance of 1,167 channels can give low-variance/noisy channels too much influence. This is a result for this specific preprocessing and kernel configuration, not a general rule against spectral scaling.

## Patient-specific prediction example

The notebook refits PCA 20 + RBF ARD on the training patients for patient 1887's original validation fold, then shows the predictive distribution and threshold-exceedance probabilities for that held-out sample and the other patients in the same fold. Patient 1887 is IDH-mutant, but the GP mean is 0.556, just above the 0.5 boundary. Its approximate 95% predictive interval is broad, from -0.528 to 1.639; the corresponding regression-derived probabilities are 0.460 mutant and 0.540 wildtype. This displays the ambiguity without turning the regression score into a confident class probability.

A further panel traces both threshold-exceedance probabilities along a straight interpolation in the 20-dimensional PCA feature space from the closest training patient (using the learned ARD length scales) to patient 1887. This is a local slice through the fitted model, not an observed spectrum or a Raman-channel axis. The x-axis is the interpolation fraction. The GP's predictive density for the selected patient is also plotted with its area divided at 0.5.

## Outlier analysis

Two training-only exclusion checks are included:

1. **Class-conditional robust distance.** Within each training fold, fit PCA on the training spectra, fit a two-dimensional Minimum Covariance Determinant estimate to PC1 and PC2 separately for each target class, and flag training patients with a chi-square-tail distance below 0.01 (2 degrees of freedom). The approximate chi-square cutoff is a screening rule, not a formal significance test for these small groups.
2. **Previously reported patient 1887.** Exclude patient 1887 only when that patient belongs to a training fold. Keep the patient in the validation fold when held out, so the evaluation still measures performance on this unusual case.

The robust rule removed 54 training appearances across the five fits, representing 18 distinct patients. Its PCA 10 accuracy fell from 0.739 to 0.652 and its ROC AUC from 0.846 to 0.698. This broad flagging suggests that the small class groups and two-dimensional distributions do not support automatic deletion based on this cutoff. Removing patient 1887 from training did not improve accuracy and lowered AUC from 0.846 to 0.771. Neither check supports deleting these patients from the cohort. A high distance may reflect genuine biological heterogeneity, acquisition quality, or both; the spectra and acquisition records need review before any exclusion decision.

## Conclusions and limitations

- There is some patient-level separation, but it is not stable enough to claim reliable classification from this cohort. In the fixed-fold sweep, PCA 20 has the highest accuracy and PCA 6 the highest AUC. Nested GPR evaluation gives lower, variable results across repeat splits.
- The component sweep is non-monotonic. PCA is unsupervised and retains variance rather than necessarily retaining IDH-discriminative signal. ARD with 20 components also has many length scales relative to the 36 training patients, and at least one kernel parameter hit an optimizer bound in every fold.
- The cohort has only 46 independent patients. The thousands of spectra do not increase the independent sample size. One changed prediction moves pooled accuracy by about 2.2 percentage points, and five folds give noisy estimates.
- The initial 15-configuration sweep used the same fixed folds, so its maximum is optimistic after model selection. The follow-up adds nested tuning, three outer split seeds, and patient-bootstrap intervals conditional on the seed-42 fitted folds, but there is still no external validation.
- Mean spectra prevent spectral-level pseudoreplication but remove spatial and within-patient structure. The direct GP classifier tested below was weaker on these folds; neither it nor the regression-derived exceedance values establish calibrated clinical probabilities.
- Outlier status is not evidence of data error. The distance cutoff is approximate and class-conditional; removing a biologically atypical patient could erase relevant signal.

## Approaches to test next

1. **Use nested patient-level validation.** Keep an outer patient-stratified split for final scoring, and choose the number of PCs, kernel, scaling, and threshold using only inner training folds. Repeat outer splits where feasible and report balanced accuracy, sensitivity/specificity, ROC AUC, and patient-bootstrap uncertainty. Preserve the original five-fold assignment as a reproducibility baseline.
2. **Fit a GP classifier.** A Bernoulli likelihood with a probit or logistic link models the binary outcome directly and produces class probabilities. Compare its patient-level results against GPR without selecting on the outer folds.
3. **Try supervised spectral reduction and compact baselines.** Evaluate PLS-DA, regularized logistic regression, and an RBF SVM with every feature-selection and tuning step inside the training folds. PCA's high-variance directions need not be the directions that separate IDH status.
4. **Review spectral quality and preprocessing.** Inspect candidate patients against raw spectra, replicate consistency, acquisition batch, and QC metadata. Consider domain-appropriate baseline correction, normalization, denoising, and wavelength alignment, validating every choice inside training folds. Use technical evidence to justify exclusions rather than an outlier score alone.
5. **Retain structure if it carries signal.** Compare patient-level means with entry-level aggregation or a hierarchical/multiple-instance model that accounts for multiple entries per patient while still splitting and scoring by patient.
6. **Collect an independent cohort.** Larger patient numbers and external validation are the clearest route to a credible estimate of accuracy and to checking whether any spectral separation generalizes across acquisition conditions.

## Follow-up tests of approaches 1-6

### 1. Nested patient-level validation

Ran three repeats of five-fold outer patient-stratified validation (seeds 42, 101, and 202). Each outer training set used three-fold inner validation to choose among 6, 8, 12, or 20 PCs; isotropic or ARD RBF; raw or channel-standardized spectra; and regression cutoffs of 0.4, 0.5, or 0.6. PCA, scaling, and GP fitting were refit within every inner and outer training fold. Optimizer restarts were set to zero for this larger grid.

| Outer-repeat seed | Accuracy | Balanced accuracy | ROC AUC |
|---:|---:|---:|---:|
| 42 | 0.739 | 0.737 | 0.855 |
| 101 | 0.717 | 0.710 | 0.789 |
| 202 | 0.696 | 0.693 | 0.741 |
| Mean (SD) | 0.717 (0.022) | 0.713 (0.022) | 0.795 (0.057) |

The selected component count, scaling, kernel, and threshold varied across outer folds. A stratified bootstrap over the seed-42 out-of-fold patient predictions gave 95% intervals of 0.609-0.870 for accuracy, 0.606-0.865 for balanced accuracy, and 0.729-0.952 for AUC. These intervals condition on the fitted folds and do not include uncertainty from refitting the full model-selection procedure.

### 2. Gaussian process classifier

Compared native Bernoulli GP classification probabilities using PCA and ARD RBF on the original patient folds. The best tested setting was PCA 8; it did not match the regression model's strongest fixed-fold metrics.

| Model | Accuracy | Balanced accuracy | ROC AUC | Brier score |
|---|---:|---:|---:|---:|
| GPC, PCA 6 | 0.609 | 0.610 | 0.670 | 0.234 |
| GPC, PCA 8 | 0.652 | 0.650 | 0.740 | 0.206 |
| GPC, PCA 10 | 0.587 | 0.586 | 0.606 | 0.265 |
| GPC, PCA 20 | 0.565 | 0.570 | 0.625 | 0.239 |

These are classifier probabilities from the GP classifier, unlike the Gaussian threshold-exceedance values from the regression models. This comparison held the kernel family fixed and was not a nested search over GPC choices.

### 3. Supervised reduction and compact baselines

Used inner patient folds to select PLS-DA components, L2-logistic regularization, or PCA dimension and regularization for an RBF SVM. The outer evaluation used the seed-42 patient folds.

| Model | Accuracy | Balanced accuracy | ROC AUC | Brier score |
|---|---:|---:|---:|---:|
| PLS-DA | 0.804 | 0.793 | 0.846 | — |
| L2 logistic regression | 0.804 | 0.793 | 0.850 | 0.153 |
| RBF SVM | 0.783 | 0.781 | 0.832 | — |

PLS-DA and logistic regression matched the fixed-sweep PCA 20 GPR accuracy on this split, while nested GPR had a slightly higher AUC on seed 42 (0.855). Logistic regression's Brier score is based on its native probabilities; the PLS and SVM outputs are not probabilities. These baselines were nested on one outer repeat only, so they need repeated outer evaluation before ranking them.

### 4. Spectral QC and preprocessing

The file contains 59 sample entries from 46 patients: 37 patients have one entry, six have two, two have three, and one has four. All individual spectra already have unit L2 norm. For the nine patients with repeated entries, mean pairwise cosine similarity between entry-mean spectra ranged from 0.899 to 0.998. This is a limited consistency check, not a technical QC verdict.

Using PCA 8 + RBF ARD on the original folds, re-normalizing each patient mean to unit L2 retained 0.739 accuracy but reduced AUC from 0.840 to 0.773. SNV on the patient means gave 0.587 accuracy and 0.690 AUC. The input is already processed and carries no wavelength calibration, acquisition-batch/date, or QC-control metadata; baseline correction, artifact review against raw spectra, derivative smoothing, and cross-instrument alignment could not be tested from this file. SNV and patient-mean renormalization are limited sensitivity checks, not endorsed preprocessing choices.

### 5. Patient means and entry-level structure

Compared patient-level pooled spectra, equal weighting of entry means, a patient mean augmented with within-patient spectral standard deviation, and an entry-level GPR whose held-out entry scores were averaged to one score per patient.

| Representation | Accuracy | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|
| Pooled patient mean | 0.739 | 0.730 | 0.840 |
| Equal mean of entry means | 0.717 | 0.702 | 0.779 |
| Patient mean + within-patient spectral SD | 0.717 | 0.710 | 0.749 |
| Entry-level GPR, average predictions to patient | 0.761 | 0.746 | 0.848 |

All models used patient-held-out folds and PCA 8 + RBF ARD. The entry-level fit showed a small apparent gain, but repeated-entry patients contributed multiple training rows and therefore more influence; scikit-learn's GPR has no sample-weight argument. Only nine patients have repeated entries, and this comparison was not nested. The result is not enough to conclude that entry-level training is better. The patient-mean-plus-SD model did not help.

### 6. External cohort

This could not be scored because the workspace contains no independent cohort. A potentially relevant independent study reports 38 fresh glioma biopsies, but its Raman acquisition and preprocessing differ from this FFPE APOLLO data, and its data availability statement says spectra are available from the corresponding author on reasonable request. No external spectra were available locally, so no external validation was run. See [Riva et al., *Cancers* (2021)](https://doi.org/10.3390/cancers13164196).

### Overall follow-up conclusion

PCA 8 was not a poor test, and supervised compact models were competitive. However, performance moves with the split and model choice: repeated nested GPR averaged 0.717 accuracy, while PLS-DA and logistic reached 0.804 on only the seed-42 outer folds. The preprocessing and entry-structure checks do not establish a robust improvement, and external validation remains unavailable. Treat all results as exploratory until a larger, independently acquired patient cohort is evaluated with the same patient-level protocol.
