import os
import pytest

from importlib import import_module

def test_package_importable():
    # ensure package imports cleanly
    mod = import_module('rrs3c.model')
    assert hasattr(mod, 'rrs_model_3C')

def test_data_files_present_or_skip():
    # the core model requires data files in data/ — skip the test if missing
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    required = [
        os.path.join(data_folder, 'vars_aph_v2.npz'),
        os.path.join(data_folder, 'abs_scat_seawater_20d_35PSU_20230922_short.txt')
    ]
    for f in required:
        if not os.path.exists(f):
            pytest.skip(f"Required data file missing: {f}")

    # If files are present, at least construct the model
    from rrs3c.model import rrs_model_3C
    model = rrs3c.model.rrs_model_3C(data_folder=data_folder)
    assert hasattr(model, 'fit_LtEs')
