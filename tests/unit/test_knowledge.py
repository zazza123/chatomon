import pytest

from chatomon import ChatomonConfigError, Knowledge


def test_render_wraps_the_text():
    assert Knowledge("Open 9-18.").render() == "<knowledge>\nOpen 9-18.\n</knowledge>"


def test_render_includes_the_title():
    knowledge = Knowledge("Open 9-18.", title="Opening hours")
    assert knowledge.render() == '<knowledge title="Opening hours">\nOpen 9-18.\n</knowledge>'


def test_empty_text_is_rejected():
    with pytest.raises(ChatomonConfigError, match="empty"):
        Knowledge("   ")


def test_from_file_reads_the_text(tmp_path):
    path = tmp_path / "faq.md"
    path.write_text("Returns within 30 days.", encoding="utf-8")
    knowledge = Knowledge.from_file(path, title="FAQ")
    assert knowledge.text == "Returns within 30 days."
    assert knowledge.title == "FAQ"
