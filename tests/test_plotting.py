import matplotlib

from plotting import configure_matplotlib_fonts


def test_font_configuration_keeps_minus_signs_readable():
    configure_matplotlib_fonts()
    assert matplotlib.rcParams["axes.unicode_minus"] is False
