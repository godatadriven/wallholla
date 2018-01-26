import pytest

import walholla.runparser as runparser
from walholla.runparser import Run


@pytest.fixture(scope='module')
def config():
    test_file = "../runs/example_run/run.yml"
    return runparser.load_config(test_file)


def test_load_config(config):
    # Just check that is does not crash
    assert config is not None


def test_build_model(config):
    model = runparser.build_model(config)
    pass


def test_build_optimiser(config):
    optimiser = runparser.build_optimiser(config)
    pass


def test_data_loader(config):
    x, y = runparser.build_dataset(config)

    assert(x.shape == (10000, 2))
    assert(y.shape == (10000, 1))


def test_build_run(config):
    run = Run(config)
    run.execute()
    pass