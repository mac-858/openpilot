"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import io
import os
import struct

import pytest

import openpilot.sunnypilot.models.helpers as helpers
import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
from openpilot.sunnypilot.modeld_v2.compile_modeld import POLICY_INPUTS, WARP_INPUTS
from openpilot.sunnypilot.modeld_v2.modeld import _find_driving_pkl, _load_jits, select_devices
from openpilot.sunnypilot.modeld_v2.tests.conftest import DummyModel, DummyBundle, ARCHETYPES, CAM_W, CAM_H, \
  SPLIT_VISION_INPUT_SHAPES, make_pkl_data, write_pkl

ModelState = modeld_module.ModelState
ARCHETYPE_NAMES = list(ARCHETYPES.keys())


# Pkl discovery

class TestFindDrivingPkl:
  def test_returns_none_when_no_bundle(self):
    assert _find_driving_pkl(None) is None

  def test_returns_none_when_no_models(self):
    bundle = DummyBundle(models=[])
    assert _find_driving_pkl(bundle) is None

  def test_returns_none_when_pkl_not_on_disk(self):
    bundle = DummyBundle(models=[DummyModel('chunked', 'driving_fof_tinygrad.pkl')])
    assert _find_driving_pkl(bundle) is None

  def test_finds_pkl_by_artifact_name(self, tmp_path, monkeypatch):
    (tmp_path / 'driving_fof_tinygrad.pkl').write_bytes(b'fake')
    from openpilot.system.hardware import hw
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))

    bundle = DummyBundle(models=[DummyModel('chunked', 'driving_fof_tinygrad.pkl')])
    result = _find_driving_pkl(bundle)
    assert result is not None
    assert 'driving_fof_tinygrad.pkl' in result


# Pkl format: catalog artifacts are dump_oob streams, a plain pickle (older local compile output) still loads

class TestLoadJits:
  @pytest.mark.parametrize("oob", [True, False])
  def test_loads_both_formats(self, tmp_path, oob):
    archetype = ARCHETYPES['vision_policy_split']
    pkl_path = write_pkl(tmp_path, archetype, oob=oob)
    jits = _load_jits(str(pkl_path))
    assert jits['metadata'] == make_pkl_data(archetype)['metadata']
    assert (CAM_W, CAM_H) in jits and 'run_policy' in jits

  def test_loads_chunked_oob(self, tmp_path):
    from openpilot.common.file_chunker import chunk_file, get_chunk_targets
    archetype = ARCHETYPES['supercombo_non20hz']
    pkl_path = write_pkl(tmp_path, archetype, oob=True)
    chunk_file(str(pkl_path), get_chunk_targets(str(pkl_path), 1 + 2 * 45 * 1024 * 1024))  # force 3 chunk files
    assert not pkl_path.exists()
    assert _load_jits(str(pkl_path))['metadata'] == make_pkl_data(archetype)['metadata']

  @pytest.mark.parametrize("cut", [4, 8, 20, -1])
  def test_truncated_oob_fails_loudly(self, tmp_path, cut):
    """A short read anywhere (header, opcodes, buffer payload) must raise, never zero-fill."""
    archetype = ARCHETYPES['supercombo_non20hz']
    pkl_path = write_pkl(tmp_path, archetype, oob=True)
    data = pkl_path.read_bytes()
    pkl_path.write_bytes(data[:cut] if cut > 0 else data[:-1])
    with pytest.raises((EOFError, ValueError, struct.error)):
      _load_jits(str(pkl_path))

  def test_truncated_buffer_payload_fails_loudly(self):
    """The out-of-band buffers hold the weights: a truncated one must raise rather than come back zeroed."""
    import numpy as np
    weights = np.arange(1024, dtype=np.float32)
    buf = io.BytesIO()
    dump_oob({'w': np.ascontiguousarray(weights)}, buf)
    full = buf.getvalue()
    assert load_oob(io.BufferedReader(io.BytesIO(full)))['w'].tolist() == weights.tolist()
    with pytest.raises(EOFError, match="truncated"):
      load_oob(io.BufferedReader(io.BytesIO(full[:-100])))

  def test_truncated_chunk_fails_loudly(self, tmp_path):
    from openpilot.common.file_chunker import chunk_file, get_chunk_name, get_chunk_targets
    archetype = ARCHETYPES['supercombo_non20hz']
    pkl_path = write_pkl(tmp_path, archetype, oob=True)
    chunk_file(str(pkl_path), get_chunk_targets(str(pkl_path), 1 + 45 * 1024 * 1024))  # 2 chunk files, data in the first
    first = get_chunk_name(str(pkl_path), 0, 2)
    with open(first, 'r+b') as f:
      f.truncate(os.path.getsize(first) - 5)
    with pytest.raises((EOFError, ValueError, struct.error)):
      _load_jits(str(pkl_path))


# Device selection: the same USB GPU gate that picks the catalog picks the model device

class TestSelectDevices:
  def test_small_model_uses_hardware_default(self, monkeypatch):
    monkeypatch.setattr(modeld_module, 'TICI', True)
    assert select_devices(False, {}) == ('QCOM', 'QCOM')
    monkeypatch.setattr(modeld_module, 'TICI', False)
    assert select_devices(False, {}) == ('CPU', 'CPU')

  def test_big_model_runs_on_amd_with_pkl_warp_device(self, monkeypatch):
    monkeypatch.setattr(modeld_module, 'TICI', True)
    assert select_devices(True, {'warp_dev': 'QCOM'}) == ('QCOM', 'AMD')
    assert select_devices(True, {}) == ('QCOM', 'AMD')

  def test_gate_off_means_no_chestnut_anywhere(self, monkeypatch):
    monkeypatch.setattr(helpers, 'CHESTNUT_MODELS_ENABLED', False)
    monkeypatch.setattr(helpers, 'usbgpu_present', lambda: True)
    assert helpers.chestnut_present() is False
    assert helpers.get_active_source() == "qcom"
    monkeypatch.setattr(helpers, 'CHESTNUT_MODELS_ENABLED', True)
    assert helpers.chestnut_present() is True
    assert helpers.get_active_source() == "chestnut"


# Init — assertion guard

class TestModelStateCombinedInit:
  def test_asserts_when_no_pkl(self, monkeypatch):
    bundle = DummyBundle(models=[], is_20hz=True)
    monkeypatch.setattr(helpers, 'get_active_bundle', lambda params=None, *, chestnut=None: bundle)
    monkeypatch.setattr(modeld_module, 'get_active_bundle', lambda params=None, *, chestnut=None: bundle)
    with pytest.raises(AssertionError, match="No driving pkl found"):
      ModelState(cam_w=CAM_W, cam_h=CAM_H)


# JIT calling protocol: kwargs must match what the published catalog pkls were compiled with

class TestCatalogProtocol:
  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_queue_keys_match_compiled_jit_inputs(self, archetype_name, model_state_factory):
    state = model_state_factory(ARCHETYPES[archetype_name])
    assert set(state.input_queues.keys()) == set(WARP_INPUTS) | set(POLICY_INPUTS)
    assert 'packed_npy_inputs' in state.input_queues
    assert state.run_policy is not None and state.warp is not None

  def test_supercombo_packs_prev_feat(self, model_state_factory):
    state = model_state_factory(ARCHETYPES['supercombo_non20hz'])
    assert 'prev_feat' in state.numpy_inputs
    assert state.numpy_inputs['prev_feat'].shape == (1, 512)

  def test_split_queue_keys_work_with_desire_key(self):
    from openpilot.sunnypilot.modeld_v2.compile_modeld import derive_frame_skip, make_split_input_queues

    policy_shapes_desire = {'features_buffer': (1, 25, 512), 'desire': (1, 25, 8), 'traffic_convention': (1, 2)}
    frame_skip = derive_frame_skip(SPLIT_VISION_INPUT_SHAPES, policy_shapes_desire)
    queues, npy = make_split_input_queues(SPLIT_VISION_INPUT_SHAPES, policy_shapes_desire, frame_skip, device='NPY')

    assert 'desire_q' in queues
    assert 'desire' in npy
    assert 'img_q' in queues
    assert 'feat_q' in queues

  def test_split_vision_input_names(self, model_state_factory):
    state = model_state_factory(ARCHETYPES['vision_policy_split'])
    assert state.vision_input_names == ['img', 'big_img']

  def test_split_output_slices_preserved(self, model_state_factory):
    arch = ARCHETYPES['vision_policy_split']
    state = model_state_factory(arch)
    assert state.vision_output_slices == arch.metadata_structure['vision']['output_slices']
    assert state.policy_output_slices == arch.metadata_structure['policy']['output_slices']


class TestModelTypeDetection:
  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_combined_model_type(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state._combined_model_type == arch.expected_model_type

  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_constants_class(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert type(state.constants) is arch.expected_constants_class

  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_parser_module(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert type(state.parser).__module__.endswith(arch.expected_parser_module)

  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_desire_key(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state.desire_key == arch.expected_desire_key

  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_npy_contains_transforms_and_desire(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state.numpy_inputs['tfm'].shape == (3, 3)
    assert state.numpy_inputs['big_tfm'].shape == (3, 3)
    assert arch.expected_desire_key in state.numpy_inputs

  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_frame_buf_params_per_vision_input(self, archetype_name, model_state_factory):
    state = model_state_factory(ARCHETYPES[archetype_name])
    for name in state.vision_input_names:
      assert len(state.frame_buf_params[name]) >= 4

  @pytest.mark.parametrize("archetype_name", ARCHETYPE_NAMES)
  def test_bundle_overrides_and_generation(self, archetype_name, model_state_factory):
    state = model_state_factory(ARCHETYPES[archetype_name])
    assert state.LAT_SMOOTH_SECONDS == 0.1
    assert state.LONG_SMOOTH_SECONDS == 0.3
    assert state.generation == 10
    assert state.mlsim is False
    assert state.chestnut is False


class TestMlsimProperty:
  def test_mlsim_true_for_gen11(self, tmp_path, monkeypatch, patch_modeld):
    from openpilot.system.hardware import hw

    arch = ARCHETYPES['supercombo_non20hz']
    write_pkl(tmp_path, arch)
    bundle = DummyBundle(models=arch.model_stubs, is_20hz=arch.is_20hz, generation=11)
    patch_modeld(bundle)
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))

    state = ModelState(cam_w=CAM_W, cam_h=CAM_H)
    assert state.mlsim is True


class TestCrossArchetypeMismatch:
  def test_wrong_is_20hz_changes_constants(self, tmp_path, monkeypatch, patch_modeld):
    from openpilot.system.hardware import hw
    from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

    arch = ARCHETYPES['vision_policy_split']
    write_pkl(tmp_path, arch)
    bundle = DummyBundle(models=arch.model_stubs, is_20hz=False)
    patch_modeld(bundle)
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))

    state = ModelState(cam_w=CAM_W, cam_h=CAM_H)
    assert type(state.constants) is ModelConstants
