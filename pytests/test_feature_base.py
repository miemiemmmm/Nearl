import pytest

from nearl.features import Mass
from nearl.featurizer import Featurizer


def test_padding_setter_coerces_to_float():
    feat = Mass()
    feat.padding = 5
    assert feat.padding == 5.0
    assert isinstance(feat.padding, float)


def test_padding_setter_rejects_none():
    feat = Mass()
    with pytest.raises(TypeError):
        feat.padding = None


def test_frame_offset_setter_coerces_to_int():
    feat = Mass()
    feat.frame_offset = 3.7
    assert feat.frame_offset == 3
    assert isinstance(feat.frame_offset, int)


def test_hook_defaults_padding_to_cutoff():
    featurizer = Featurizer(
        {"dimensions": 32, "lengths": 16, "cutoff": 3.5, "sigma": 1.5}
    )
    feat = Mass(outkey="mass")
    featurizer.register_feature(feat)
    assert feat.padding == 3.5


def test_hook_without_cutoff_currently_raises_type_error():
    # Known pre-existing limitation, pinned rather than fixed here: hook()
    # defaults padding to cutoff unconditionally, which crashes if cutoff
    # was never provided anywhere (neither on the Feature nor the Featurizer).
    featurizer = Featurizer({"dimensions": 32, "lengths": 16})
    feat = Mass(outkey="mass")
    with pytest.raises(TypeError):
        featurizer.register_feature(feat)
