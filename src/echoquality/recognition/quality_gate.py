"""Public Quality Gate API.

The implementation is hierarchical and uses late fusion; this module keeps the
scientifically meaningful package boundary requested by the project design.
"""

from echoquality.recognition.hierarchical_gate import (  # noqa: F401
    NestedValidationResult,
    fit_final_gate,
    infer_feature_columns_by_level,
    load_gate,
    nested_group_validation,
    save_gate,
)
