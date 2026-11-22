"""Quick tour of subjectwise on the synthetic cohort (runs in a few seconds).

Run it with ``python examples/quickstart.py``. To use your own data, replace the first three lines
with ``df = sw.read_table("my_features.csv")`` and list your feature columns.
"""

import subjectwise as sw

df = sw.toy_cohort(n_subjects=8, trials_per_level=5, seed=0)
features, log_features = sw.toy_features()  # amp_* are positive and multiplicative -> log scale

# 1. Is the table sound? (missing classes, tiny or duplicated subjects, imbalance, ...)
print(sw.audit(df, features, label="level", group="subject").format())

# 2. How optimistic is a random split compared with holding out whole subjects?
schemes = sw.compare_schemes(
    df, features, "level", "subject", models=("lda",), log_features=log_features
)
print(
    schemes[["scheme", "normalize", "mean_accuracy", "sd_accuracy"]].round(3).to_string(index=False)
)
print(sw.leakage_gap(schemes).round(3).to_string(index=False))

# 3. Is the honest number above chance? Shuffle labels inside each subject and repeat everything.
perm = sw.permutation_test(
    df,
    features,
    "level",
    "subject",
    normalize="within_group",
    log_features=log_features,
    n_perm=100,
)
print(perm.summary())

# 4. Which subjects drive it?
print(sw.leave_group_out(df, features, "level", "subject", log_features=log_features).round(3))

# 5. Simulate new subjects from this pilot and ask how many a study would need.
generator = sw.HierarchicalGenerator.fit(df, features, "subject", "x", log_features=log_features)
power = sw.power_curve(
    generator.scale_effect(0.15),  # pilots overstate effects: plan for a weaker one
    sw.level_x_stats(df, "level", "x"),
    n_subjects_list=(4, 8, 12),
    trials_per_level_list=(5,),
    n_reps=20,
    normalize="within_group",
)
print(
    power.summary[["n_subjects", "null_threshold", "power_calibrated"]]
    .round(2)
    .to_string(index=False)
)
