from __future__ import annotations

import importlib.metadata

import quantity as m


def test_version():
    assert importlib.metadata.version("quantity") == m.__version__
