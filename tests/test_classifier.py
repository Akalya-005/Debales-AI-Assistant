from app.classifier import Intent, classify_query


def test_debales_query_routes_to_rag():
    result = classify_query("What does Debales AI automate?")
    assert result.intent == Intent.DEBALES


def test_external_query_routes_to_search():
    result = classify_query("What is AI in logistics?")
    assert result.intent == Intent.EXTERNAL


def test_mixed_query_routes_to_both():
    result = classify_query("Compare Debales AI with general logistics automation.")
    assert result.intent == Intent.MIXED


def test_empty_query_is_unknown():
    result = classify_query("")
    assert result.intent == Intent.UNKNOWN


def test_vague_query_is_unknown():
    result = classify_query("hi")
    assert result.intent == Intent.UNKNOWN
