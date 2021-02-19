import pkg_resources
import pytest

import stan
import stan.plugins

program_code = "parameters {real y;} model {y ~ normal(0,1);}"


class DummyPlugin(stan.plugins.PluginBase):
    def on_post_fit(self, fit):
        """Do nothing other than print a string."""
        print("In DummyPlugin `on_post_fit`.")
        return fit


class MockEntryPoint:
    @staticmethod
    def load():
        return DummyPlugin


def mock_iter_entry_points(group):
    return iter([MockEntryPoint])


@pytest.fixture(scope="module")
def normal_posterior():
    return stan.build(program_code)


def test_get_plugins(monkeypatch):

    monkeypatch.setattr(pkg_resources, "iter_entry_points", mock_iter_entry_points)

    entry_points = stan.plugins.get_plugins()
    Plugin = next(entry_points).load()
    assert isinstance(Plugin(), stan.plugins.PluginBase)


def test_dummy_plugin(monkeypatch, capsys, normal_posterior):

    monkeypatch.setattr(pkg_resources, "iter_entry_points", mock_iter_entry_points)

    fit = normal_posterior.sample(stepsize=0.001)
    assert fit is not None and "y" in fit

    captured = capsys.readouterr()
    assert "In DummyPlugin" in captured.out


class OtherDummyPlugin(stan.plugins.PluginBase):
    def on_post_fit(self, fit):
        """Do nothing other than print a string."""
        print("In OtherDummyPlugin `on_post_fit`.")
        return fit


class OtherMockEntryPoint:
    @staticmethod
    def load():
        return OtherDummyPlugin


def test_two_plugins(monkeypatch, capsys, normal_posterior):
    """Make sure that both plugins are used."""

    def mock_iter_entry_points(group):
        return iter([MockEntryPoint, OtherMockEntryPoint])

    monkeypatch.setattr(pkg_resources, "iter_entry_points", mock_iter_entry_points)

    fit = normal_posterior.sample(stepsize=0.001)
    assert fit is not None and "y" in fit

    captured = capsys.readouterr()
    assert "In DummyPlugin" in captured.out
    assert "In OtherDummyPlugin" in captured.out
