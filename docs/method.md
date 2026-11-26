# Methods

Short notes on what each part of `subjectwise` computes and why. Everything here is implemented in
`src/subjectwise/` and covered by tests.

## 1. Leave-one-subject-out and the number you report

Rows from one subject share electrode placement, anatomy, effort and dozens of other nuisance
factors. With a random split (`kfold`) rows of every subject land on both sides of the boundary and
the model can score well by recognising the person. `loso` holds out one whole subject at a time,
and `group_kfold` holds out several; `evaluate` asserts that no subject appears in both train and
test.

The headline is the **mean of per-subject accuracies**, with their sample standard deviation
(`ddof=1`) across subjects. Pooling rows instead lets a subject with 80 trials count four times as
much as one with 20. The pooled balanced accuracy and macro-F1 are reported next to it.

## 2. Within-subject z-score

For every subject and feature: `z = (v - mean_s) / sd_s` (sample SD). Features declared
multiplicative (`log_features`: RMS, power, ratios) are put on a log scale first, so that an
electrode gain that multiplies the signal by 5 becomes an additive shift that the z-score removes.
Zeros are clipped to `1e-12`, negative values raise, constant features and one-row subjects become
0, NaN stays NaN.

The statistics come from the subject's own rows and **never from labels**, so a leave-one-subject-out
estimate stays leakage-free. The price is that it is *transductive*: at deployment a new person
needs an unlabelled calibration block that covers the range of the task, otherwise the mean and SD
are biased. In the demo this normalisation is what takes LDA from 57% to 91%, which is also a
warning: a gain this large says the raw features were dominated by between-person scale.

## 3. Permutation test

"Chance = 1/K" holds only for balanced classes and an uninformative model. The honest reference is a
null distribution: shuffle the labels, run the very same pipeline, and count how often chance
reaches the observed score.

* Labels are shuffled **inside every subject**. Each subject keeps its class composition (and its
  majority-class baseline); only the link between features and labels is destroyed.
* `p = (1 + #{null >= observed}) / (1 + n_perm)`, so it is never exactly zero.
* The normalisation ignores labels, so it is computed once and not repeated per permutation.
* Every permutation gets its own seed, drawn up front from the master seed, so the null is
  identical for any `n_jobs`.

## 4. Correlation: three views

`correlation_report(df, x, y, group)` returns

1. **pooled**: all rows together (mixes between- and within-subject variation);
2. **within**: after centring `x` and `y` inside each subject;
3. **subject-level**: one Spearman/Pearson coefficient per subject, Fisher z-transformed, tested
   against zero with a t-test over subjects. Its `n` is the number of subjects, which is the honest
   number of independent observations.

When subjects differ in scale the pooled and within-subject coefficients can disagree even in sign
(Simpson's paradox); `tests/test_correlation.py` builds that case. Pearson or Spearman is chosen by
Shapiro-Wilk on both variables.

## 5. Audit codes

| code | severity | meaning |
|---|---|---|
| `non_finite` | error | NaN or inf in a feature |
| `too_few_groups` | error | fewer than 2 subjects |
| `single_class` | error | fewer than 2 classes |
| `two_groups` | warning | LOSO is then one split per direction |
| `tiny_group` | warning | a subject with fewer rows than `min_rows_per_group` |
| `missing_class_in_group` | warning | a subject never performed a class |
| `imbalanced_classes` | warning | majority-class rate more than 5 points above `1/K` |
| `duplicate_groups` | warning | two subjects with identical feature rows (a copied recording) |
| `constant_feature` | warning | a feature that never changes |

`duplicate_groups` compares an order-independent fingerprint of each subject's rows (rounded to six
decimals, signed zeros normalised). A copy of the held-out subject that stays in the training set is
one of the most common reasons for a result that is too good.

## 6. Simulated subjects (`HierarchicalGenerator`)

Per feature `j` (log scale for multiplicative features), subject `s`, trial `i`:

    y[i, j] = a[s, j] + b[s, j] * (x[i] - x0) + e[i, j]

* `(a, b)` are fitted by least squares for each pilot subject (`x0` = mean of `x`, so intercepts and
  slopes decouple). Residuals are inflated by `sqrt(n / (n - 2))` for the two fitted parameters.
* A *new* subject is `mu + w @ D`, where `D` holds the pilot subjects' deviations from the mean and
  `w ~ N(0, s2 / (k - 1))`: a Gaussian with the empirical between-subject covariance of the `k`
  pilot subjects, which keeps the correlations between intercepts and slopes of different features.
* `s2 = (k - 1) / chi2(k - 1)` draws the between-subject variance itself from its sampling
  distribution, so that the uncertainty of a small pilot appears in the simulated studies instead of
  being hidden. `spread_interval()` reports the 95% interval of that factor (for 8 pilot subjects:
  roughly 0.66 to 2.0).
* `e` is multivariate Gaussian. The residual **correlation** matrix is estimated with Ledoit-Wolf
  shrinkage on *standardised* residuals (shrinking raw covariances towards a scaled identity would
  drag features of very different scales towards a common variance), then rescaled by each
  feature's SD and sampled through a Cholesky factor.
* `scale_effect(f)` multiplies the mean slopes and their between-subject spread by `f`. `f = 0`
  gives a null generator in which features carry no information about `x`.

`calibration_check` fits the generator on all subjects but one, simulates `n_sim` subjects, and
reports the percentile of the real held-out subject's mean for every feature. About 90% of the
points should fall in `[5, 95]`; far fewer means the generator is too narrow, far more that it is
too wide.

Limitations: Gaussian, linear in `x`, and it can only reproduce what the pilot shows. Per-subject
estimates carry sampling noise, which inflates the estimated spread (the generator errs on the
conservative side).

## 7. Power by simulation

For every design (number of subjects x trials per class) `power_curve` simulates `n_reps` complete
studies, runs the real LOSO pipeline on each, and also simulates `null_reps` studies with
`scale_effect(0)`. Three decision rules are reported:

* **calibrated** (recommended): a study succeeds when its mean per-subject accuracy exceeds the
  `1 - alpha` quantile of the null studies of the same design and pipeline (`null_threshold`). The
  false-positive rate is at most `alpha` by construction, whatever the model.
* a one-sided **t-test** of the per-subject accuracies against `1/K`;
* a **binomial** test of the pooled number of correct predictions against `1/K`.

The two tests are shown because people use them, but in this simulation both are somewhat liberal
(see the README for the measured rates), so the power they report is optimistic. We did not chase
the mechanism: LOSO folds share most of their training data, and accuracies from a handful of
subjects are discrete and bounded, both plausible contributors.
`PowerResult.false_positive_rates()` measures the rates for the pipeline at hand.

## 8. Tuning inside the training subjects (nested validation)

The models run with fixed settings unless you ask otherwise. Choosing `C` or `gamma` on the folds
that report the accuracy is optimistic, because the settings won *since* they did well on those
subjects; choosing them with a random inner split brings back the subject leakage that LOSO
removes. `evaluate(..., tune=True)` nests the choice:

1. The held-out subject (or subjects) of the outer fold is set aside first.
2. On the remaining **training subjects** an inner cross-validation grouped by subject
   (`inner_scheme="loso"`, or `"group_kfold"` with `inner_splits` folds) scores every candidate of
   the grid by the mean, over inner held-out subjects, of their accuracy: the same headline as
   everywhere else.
3. The best candidate is refitted on all the training subjects and predicts the held-out subject
   once.

Ties go to the earliest candidate, and every default grid starts with the fixed default settings,
so they are kept unless something is clearly better on the training subjects. The default grids
are `svm_rbf`: `C` in {1, 0.1, 10, 100} x `gamma` in {s, s/10, 10 s}, with `s = 1 / n_features`
(what scikit-learn's `"scale"` is after the standardisation every model starts with);
`svm_linear`: `C` in {1, 0.01, 0.1, 10}; `rf`: `min_samples_leaf` in {1, 3, 8} x `max_features`
in {`"sqrt"`, 0.5}. Any other model, `lda` included, is tuned with an explicit
`grid={"solver": ["svd", "lsqr"]}`. Only the grouped outer schemes can be tuned, and at least 3
subjects are needed (2 training subjects for the inner loop).

What is reported. `EvalResult.tuning` has one row per outer fold: the chosen settings
(`param_*`), their inner score and the outer accuracy. `mean_accuracy` stays the outer one, on
subjects the choice never saw: that is the number to report. `inner_accuracy` is the mean inner
score of the chosen settings (each held-out subject weighted once) and `optimism` is
`inner_accuracy - mean_accuracy`. The two are not an estimate of the same thing: the inner score is
what the choice was made on, and it is flattered by having picked the best of several candidates.
On the demo cohort (seed 0, SVM with RBF kernel, LOSO + within-subject z-score) the fixed settings
give 90.0%, the nested procedure 90.0% (outer) and 91.6% (inner): the tuning found nothing to gain,
and the inner score overstated it by 1.6 points. On pure noise, picking the best of 12 candidates
(`tests/test_tuning.py`: 10 cohorts of 6 subjects, 12 rows each, 3 classes) gives an inner score of
39.9% and an outer one of 30.0%, against a chance level of 33%: a 10-point optimism that the outer
score does not carry.

The permutation test (`permutation_test(..., tune=True)`) takes its observed score from the nested
procedure and **repeats the tuning inside every permutation**, on the shuffled labels, so the null
belongs to the procedure that produced the observed score. In pure-noise checks (6 subjects, 12
rows each, SVM with RBF kernel, 80 shuffles, 3 cohorts per design) the tuned and the untuned nulls
were close: with balanced classes the mean was 33.8% against 33.5% and the 95th percentile 44.1%
against 44.5%; with classes in 60/25/15 proportions, 54.6% against 54.5% and 57.0% against 58.8%.
So repeating the tuning is not a large correction in those cases; it makes the p-value answer for
the procedure that was run, and it is what costs time. On 40 noise cohorts of 5 subjects (19
shuffles each) the tuned test with `svm_linear` called 2 significant at p <= 0.1, where about 4
are expected
(`tests/test_tuning.py::test_the_tuned_permutation_test_is_not_liberal_on_pure_noise`).

Cost: a nested evaluation fits `outer folds x inner folds x candidates` models, e.g. 8 x 7 x 12 =
672 for the demo's SVM (about 1.5 s), and a permutation test repeats that `n_perm + 1` times. Use
`n_jobs`, fewer permutations, `inner_scheme="group_kfold"` with few folds, or a smaller `grid`;
the random forest is the slow one.

What nesting does not fix: anything chosen by looking at the outer score (which model, which
features, which normalisation, which grid) is tuning on the test subjects again. Fix those choices
before looking, or nest them too.
