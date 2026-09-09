"""Phase 0: every module imports, and unbuilt stubs fail loudly rather than silently."""

import importlib

import pytest

MODULES = [
    "figomeas.config", "figomeas.io", "figomeas.manifest", "figomeas.body",
    "figomeas.features", "figomeas.figo", "figomeas.uncertainty",
    "figomeas.borderline", "figomeas.robustness", "figomeas.synthetic",
    "figomeas.viz",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None


def test_package_exports_version():
    import figomeas
    assert figomeas.__version__


# Drop a module from this list as its phase lands. Phases 1-7 are implemented, so
# only the Phase 8 reporting entry points remain unbuilt.
@pytest.mark.parametrize(
    "module,func",
    [
        ("figomeas.robustness", "run"),
    ],
)
def test_stubs_raise_not_implemented(module, func):
    fn = getattr(importlib.import_module(module), func)
    with pytest.raises(NotImplementedError):
        fn(*([None] * fn.__code__.co_argcount))


@pytest.mark.parametrize("module,func", [
    ("figomeas.io", "load_mask"), ("figomeas.body", "reconstruct"),
    ("figomeas.features", "extract_patient"), ("figomeas.figo", "derive"),
    ("figomeas.uncertainty", "confidence_interval"),
    ("figomeas.borderline", "flag"), ("figomeas.robustness", "decimate_slices"),
    ("figomeas.synthetic", "build_demo_cohort"), ("figomeas.viz", "overlay_body"),
])
def test_implemented_entry_points_exist(module, func):
    assert callable(getattr(importlib.import_module(module), func))
