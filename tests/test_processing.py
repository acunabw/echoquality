from __future__ import annotations

from echoquality.processing.adaptive_processing import candidate_transformations


def test_processing_is_conditioned_and_does_not_claim_to_fix_missing_anatomy() -> None:
    review = candidate_transformations(["review_or_reacquisition"])
    assert [candidate.label() for candidate in review] == ["identity"]
    contrast = candidate_transformations(["global_contrast"], max_candidates=8)
    assert contrast[0].label() == "identity"
    assert any("gamma" in candidate.label() or "clahe" in candidate.label() for candidate in contrast)
