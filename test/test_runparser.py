import pytest

from walholla.runparser import build_model, load_config


def test_load_config():
    test_file = "../runs/example_run/run.yml"
    config = load_config(test_file)

    # Just check that is does not crash
    assert config is not None


def test_build_model():

    test_file = "../runs/example_run/run.yml"
    config = load_config(test_file)

    model = build_model(config)
    pass

