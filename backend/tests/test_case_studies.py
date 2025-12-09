from case_studies.loader import load_case_studies


def test_case_studies_count_and_fields():
    cases = load_case_studies()
    assert len(cases) == 5
    for cs in cases:
        assert cs.title
        assert cs.problem
        assert cs.solution
        assert cs.outcome
