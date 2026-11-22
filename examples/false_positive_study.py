"""How liberal are the classical tests when there is no effect at all?

Simulates studies from a null generator (``scale_effect(0)``: features carry no information about
the class), runs leave-one-subject-out on each, and reports the share of studies that a one-sided
t-test over subjects and a pooled binomial test call significant at 5%. A valid test gives 5%.

    python examples/false_positive_study.py --reps 800     # about 10 minutes
    python examples/false_positive_study.py --reps 100     # quick look, +/- 2 points of noise
"""

import argparse
import warnings

import numpy as np
import pandas as pd

import subjectwise as sw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=800, help="simulated studies per design")
    parser.add_argument("--subjects", default="4,8,12", help="numbers of subjects, comma-separated")
    parser.add_argument("--trials", default="2,5", help="trials per class, comma-separated")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    df = sw.toy_cohort(n_subjects=8, trials_per_level=5, seed=0)
    features, log_features = sw.toy_features()
    generator = sw.HierarchicalGenerator.fit(
        df, features, "subject", "x", log_features=log_features
    )
    x_stats = sw.level_x_stats(df, "level", "x")

    tables = []
    for normalize in ("within_group", "none"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = sw.power_curve(
                generator.scale_effect(0.0),
                x_stats,
                n_subjects_list=[int(v) for v in args.subjects.split(",")],
                trials_per_level_list=[int(v) for v in args.trials.split(",")],
                n_reps=args.reps,
                calibrate=False,  # on null studies "power" is the false-positive rate
                normalize=normalize,
                seed=args.seed,
            )
        table = result.summary[["trials_per_level", "n_subjects", "power_ttest", "power_binom"]]
        table = table.rename(columns={"power_ttest": "fpr_ttest", "power_binom": "fpr_binom"})
        table.insert(0, "normalize", normalize)
        tables.append(table)

    out = pd.concat(tables, ignore_index=True)
    se = np.sqrt(0.05 * 0.95 / args.reps)
    print(out.round(3).to_string(index=False))
    print(f"\nA valid 5% test gives 0.05; the Monte-Carlo standard error is about {se:.3f}.")


if __name__ == "__main__":
    main()
