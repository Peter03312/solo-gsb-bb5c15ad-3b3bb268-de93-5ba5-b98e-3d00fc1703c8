import pytest

from app.parser import FormulaError, parse_formula


def test_simple_formula():
    parsed = parse_formula("C6H12O6")
    assert parsed.composition == {"C": 6, "H": 12, "O": 6}
    assert parsed.labels == {}


def test_implicit_counts_and_repeated_elements():
    parsed = parse_formula("CH3COOH")
    assert parsed.composition == {"C": 2, "H": 4, "O": 2}


def test_isotope_label():
    parsed = parse_formula("C6H6[13C]2")
    assert parsed.composition == {"C": 6, "H": 6}
    assert parsed.labels == {("C", 13): 2}


def test_multiple_labels():
    parsed = parse_formula("C2H5[15N]O[18O]2")
    assert parsed.labels == {("N", 15): 1, ("O", 18): 2}


def test_lowercase_start_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("c6H12")
    assert excinfo.value.position == 0
    assert "uppercase" in excinfo.value.message


def test_unknown_element_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6Xx2")
    assert excinfo.value.position == 2
    assert "Xx" in excinfo.value.message


def test_unterminated_label_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6[13C")
    assert excinfo.value.position == 2
    assert "Unterminated" in excinfo.value.message


def test_malformed_label_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6[C13]")
    assert excinfo.value.position == 2
    assert "Malformed" in excinfo.value.message


def test_unknown_label_mass_reports_position():
    # 20C is not a known carbon isotope
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6[20C]")
    assert excinfo.value.position == 2
    assert "20" in excinfo.value.message


def test_unknown_element_in_label_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6[13Zz]")
    assert excinfo.value.position == 5  # points at 'Zz' inside the brackets


def test_zero_count_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C0H2")
    assert excinfo.value.position == 1
    assert "positive integer" in excinfo.value.message


def test_stray_digit_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6 12")
    assert excinfo.value.position == 2


def test_unexpected_character_reports_position():
    with pytest.raises(FormulaError) as excinfo:
        parse_formula("C6-H2")
    assert excinfo.value.position == 2
    assert "'-'" in excinfo.value.message


def test_empty_formula():
    with pytest.raises(FormulaError):
        parse_formula("")
