from personas.loader import load_personas


def test_persona_loader_returns_five():
    personas = load_personas()
    assert len(personas) == 5
    names = {p.name for p in personas}
    assert "Retirees chasing yield" in names
    assert "Self directed HNW" in names


def test_persona_key_fields_non_empty():
    personas = load_personas()
    for p in personas:
        assert p.id
        assert p.primary_goal
        assert p.key_concern
        assert p.why_private_credit_appeals
