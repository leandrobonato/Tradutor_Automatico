from src.data.preprocess import clean_text, is_valid_pair


def test_clean_text_collapses_whitespace():
    assert clean_text("  Hello   world  \n") == "Hello world"


def test_clean_text_strips_edges():
    assert clean_text("\tOlá mundo\t") == "Olá mundo"


def test_is_valid_pair_accepts_normal_sentence():
    assert is_valid_pair("How are you today?", "Como você está hoje?") is True


def test_is_valid_pair_rejects_empty():
    assert is_valid_pair("", "Como você está hoje?") is False
    assert is_valid_pair("How are you today?", "") is False


def test_is_valid_pair_rejects_too_long():
    long_text = "a" * 500
    assert is_valid_pair(long_text, "Como você está hoje?") is False


def test_is_valid_pair_rejects_unbalanced_ratio():
    assert is_valid_pair("Hi", "Este é um texto muito mais longo do que o original em inglês, claramente desbalanceado") is False
