import pytest

from app.adduct import AdductError, apply_adduct, parse_adduct
from app.parser import parse_formula


def test_parse_single_additive_fragment():
    pieces = parse_adduct("+H")
    assert len(pieces) == 1
    assert pieces[0].sign == 1
    assert pieces[0].coefficient == 1
    assert pieces[0].fragment.composition == {"H": 1}
    assert pieces[0].fragment_start == 1


def test_parse_coefficient_multiplies_whole_fragment():
    pieces = parse_adduct("+2H")
    assert pieces[0].coefficient == 2
    # the coefficient stays outside the parsed fragment
    assert pieces[0].fragment.composition == {"H": 1}
    assert pieces[0].fragment_start == 2


def test_parse_compound_adduct():
    pieces = parse_adduct("+2H-H2O")
    assert [(p.sign, p.coefficient) for p in pieces] == [(1, 2), (-1, 1)]
    assert pieces[0].fragment.composition == {"H": 1}
    assert pieces[1].fragment.composition == {"H": 2, "O": 1}
    # second fragment starts right after '-'
    assert pieces[1].fragment_start == 4


def test_parse_sodium_minus_hydrogen():
    pieces = parse_adduct("+Na-H")
    assert [p.fragment.composition for p in pieces] == [{"Na": 1}, {"H": 1}]


def test_parse_multi_element_fragment_with_coefficient():
    pieces = parse_adduct("-3H2O")
    assert pieces[0].sign == -1
    assert pieces[0].coefficient == 3
    assert pieces[0].fragment.composition == {"H": 2, "O": 1}
    assert pieces[0].fragment_start == 2


def test_parse_fragment_with_isotope_label():
    pieces = parse_adduct("+[13C]2-H")
    assert pieces[0].fragment.labels == {("C", 13): 2}
    assert pieces[1].fragment.composition == {"H": 1}


def test_empty_adduct_reports_position_zero():
    for adduct in ["", "   "]:
        with pytest.raises(AdductError) as excinfo:
            parse_adduct(adduct)
        assert excinfo.value.position == 0


def test_whitespace_reports_offending_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+ H")
    assert excinfo.value.position == 1


def test_missing_leading_sign_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("H")
    assert excinfo.value.position == 0
    assert "'+'" in excinfo.value.message


def test_sign_must_start_each_fragment():
    # fragments are delimited by signs, so the only boundary character that
    # can lack a sign is the very first one
    for adduct in ("H", "Na-H"):
        with pytest.raises(AdductError) as excinfo:
            parse_adduct(adduct)
        assert excinfo.value.position == 0
        assert "'+'" in excinfo.value.message
    # a well-formed multi-fragment string parses fully
    pieces = parse_adduct("+H-H2")
    assert [(p.sign, p.fragment.composition) for p in pieces] == [
        (1, {"H": 1}),
        (-1, {"H": 2}),
    ]


def test_empty_fragment_reports_position_after_sign():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+")
    assert excinfo.value.position == 1
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+H-")
    assert excinfo.value.position == 3


def test_zero_coefficient_reports_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+0H")
    assert excinfo.value.position == 1
    assert "positive integer" in excinfo.value.message


def test_trailing_sign_is_empty_fragment():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+H+")
    assert excinfo.value.position == 3


def test_unknown_element_in_fragment_reports_absolute_position():
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+Na-Xx")
    # second fragment starts at index 4
    assert excinfo.value.position == 4


def test_formula_error_inside_fragment_gets_absolute_position():
    # '+12C0': coefficient 12 multiplies fragment 'C0', whose zero count is
    # at relative position 1 inside the fragment (fragment starts at index 3)
    with pytest.raises(AdductError) as excinfo:
        parse_adduct("+12C0")
    assert excinfo.value.position == 4
    assert "positive integer" in excinfo.value.message
    # sanity: coefficients vs. in-fragment counts are distinguished correctly
    pieces = parse_adduct("+12H")
    assert pieces[0].coefficient == 12
    assert pieces[0].fragment.composition == {"H": 1}
    pieces = parse_adduct("+H12")
    assert pieces[0].coefficient == 1
    assert pieces[0].fragment.composition == {"H": 12}


def test_apply_simple_protonation():
    adjusted = apply_adduct(parse_formula("H2O"), parse_adduct("+H"))
    assert adjusted.composition == {"H": 3, "O": 1}


def test_apply_coefficient_and_subtraction():
    adjusted = apply_adduct(parse_formula("H2O"), parse_adduct("+2H-H2O"))
    assert adjusted.composition == {"H": 2, "O": 0}


def test_apply_sodium_adduct():
    adjusted = apply_adduct(parse_formula("CH4"), parse_adduct("+Na-H"))
    assert adjusted.composition == {"C": 1, "H": 3, "Na": 1}


def test_apply_negative_count_raises_at_offending_fragment():
    # H2O has only two H; '-3H' drives H to -1.
    with pytest.raises(AdductError) as excinfo:
        apply_adduct(parse_formula("H2O"), parse_adduct("-3H"))
    assert excinfo.value.position == 2  # first char of the 'H' fragment
    assert "negative" in excinfo.value.message


def test_negative_count_position_points_at_later_fragment():
    # +Na is fine; the following -H2 subtraction drives H below zero first
    with pytest.raises(AdductError) as excinfo:
        apply_adduct(parse_formula("H"), parse_adduct("+Na-H2"))
    assert excinfo.value.position == 4  # start of the 'H2' fragment


def test_negative_label_count_raises():
    with pytest.raises(AdductError) as excinfo:
        apply_adduct(parse_formula("[13C]2"), parse_adduct("-[13C]3"))
    assert excinfo.value.position == 1  # first char of fragment, i.e. '['
    assert "label" in excinfo.value.message


def test_zero_count_is_allowed():
    adjusted = apply_adduct(parse_formula("H2O"), parse_adduct("-H2"))
    assert adjusted.composition == {"H": 0, "O": 1}
