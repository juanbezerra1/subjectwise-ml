import pytest
from matplotlib.figure import Figure

import subjectwise as sw
from subjectwise import viz


@pytest.fixture(scope="module")
def pieces(toy):
    df, feats, logs = toy
    schemes = sw.compare_schemes(
        df, feats, "level", "subject", models=("lda", "svm_rbf"), log_features=logs
    )
    honest = sw.evaluate(df, feats, "level", "subject", normalize="within_group", log_features=logs)
    perm = sw.permutation_test(
        df, feats, "level", "subject", normalize="within_group", log_features=logs, n_perm=20
    )
    gen = sw.HierarchicalGenerator.fit(df, feats, "subject", "x", log_features=logs)
    cal = sw.calibration_check(df, feats, "subject", "x", log_features=logs, n_sim=40)
    power = sw.power_curve(
        gen.scale_effect(0.3), sw.level_x_stats(df, "level", "x"), n_subjects_list=(4, 6),
        trials_per_level_list=(2, 3), n_reps=4, null_reps=10, normalize="within_group",
    )  # fmt: skip
    return schemes, honest, perm, cal, power


def test_every_figure_is_built_and_saved_as_a_png(tmp_path, pieces):
    schemes, honest, perm, cal, power = pieces
    figures = {
        "schemes": viz.plot_scheme_comparison(schemes, chance=0.25),
        "subjects": viz.plot_subject_accuracy(honest, perm=perm),
        "subjects_chance": viz.plot_subject_accuracy(honest, chance=0.25),
        "permutation": viz.plot_permutation(perm),
        "calibration": viz.plot_calibration(cal),
        "power": viz.plot_power(power.summary, title="Power"),
        "power_t": viz.plot_power(power.summary, test="ttest"),
        "power_binom": viz.plot_power(power.summary, test="binom"),
    }
    for name, fig in figures.items():
        assert isinstance(fig, Figure)
        path = viz.save_figure(fig, tmp_path / "sub" / f"{name}.png")
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_text_is_never_drawn_in_a_series_colour(pieces):
    schemes, honest, perm, cal, power = pieces
    fig = viz.plot_scheme_comparison(schemes, chance=0.25)
    for ax in fig.axes:
        for text in ax.texts:
            assert text.get_color() not in viz.SERIES
    for fig in (
        viz.plot_permutation(perm),
        viz.plot_calibration(cal),
        viz.plot_power(power.summary),
    ):
        for ax in fig.axes:
            for text in ax.texts:
                assert text.get_color() not in viz.SERIES


def test_charts_with_two_or_more_series_have_a_legend(pieces):
    schemes, _, _, _, power = pieces
    assert viz.plot_scheme_comparison(schemes).legends
    assert viz.plot_power(power.summary).legends


def test_scheme_comparison_refuses_more_scenarios_than_colours(pieces):
    schemes = pieces[0]
    extra = schemes.assign(normalize="z")
    import pandas as pd

    with pytest.raises(ValueError, match="at most"):
        viz.plot_scheme_comparison(pd.concat([schemes, extra, extra.assign(normalize="y")]))


def test_power_plot_validates_its_arguments(pieces):
    power = pieces[4]
    with pytest.raises(ValueError, match="test must be"):
        viz.plot_power(power.summary, test="bayes")
    with pytest.raises(ValueError, match="calibrate"):
        viz.plot_power(power.summary.drop(columns="power_calibrated"))
