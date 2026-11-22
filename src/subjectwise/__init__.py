"""subjectwise: honest validation for classifiers trained on a handful of subjects.

Typical use::

    import subjectwise as sw

    df = sw.toy_cohort()
    features, log_features = sw.toy_features()
    report = sw.audit(df, features, label="level", group="subject")
    table = sw.compare_schemes(df, features, "level", "subject", log_features=log_features)
    perm = sw.permutation_test(
        df, features, "level", "subject", normalize="within_group", log_features=log_features
    )
"""

from importlib.metadata import PackageNotFoundError, version

from .checks import AuditReport, Issue, audit
from .correlation import CorrelationResult, CorrStat, correlation_report
from .cv import DEFAULT_SCENARIOS, EvalResult, compare_schemes, evaluate, leakage_gap
from .generator import HierarchicalGenerator, calibration_check, fraction_calibrated
from .models import MODEL_LABELS, MODEL_NAMES, make_model
from .normalize import within_group_zscore
from .permutation import PermutationResult, permutation_test
from .power import PowerResult, level_x_stats, power_curve
from .sensitivity import leave_group_out
from .tables import RESERVED_COLUMNS, infer_features, read_table
from .toy import toy_cohort, toy_features

try:
    __version__ = version("subjectwise-ml")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0.0.0+local"

__all__ = [
    "DEFAULT_SCENARIOS",
    "MODEL_LABELS",
    "MODEL_NAMES",
    "RESERVED_COLUMNS",
    "AuditReport",
    "CorrStat",
    "CorrelationResult",
    "EvalResult",
    "HierarchicalGenerator",
    "Issue",
    "PermutationResult",
    "PowerResult",
    "__version__",
    "audit",
    "calibration_check",
    "compare_schemes",
    "correlation_report",
    "evaluate",
    "fraction_calibrated",
    "infer_features",
    "leakage_gap",
    "leave_group_out",
    "level_x_stats",
    "make_model",
    "permutation_test",
    "power_curve",
    "read_table",
    "toy_cohort",
    "toy_features",
    "within_group_zscore",
]
