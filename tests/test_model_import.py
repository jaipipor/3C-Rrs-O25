from rrs3c import rrs_model_3C_O25


def test_model_import():
    model = rrs_model_3C_O25()

    assert callable(model.fit_LtEs)
