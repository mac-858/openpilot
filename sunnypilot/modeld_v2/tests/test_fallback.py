"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import io
import os

import pytest
import requests

from openpilot.common.file_chunker import get_chunk_name
from openpilot.selfdrive.modeld.helpers import dump_oob
from openpilot.system.hardware import hw
import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.tests.conftest import ARCHETYPES, CAM_W, CAM_H, make_pkl_data
from openpilot.sunnypilot.models.fetcher import ModelParser, ModelFetcher


@pytest.mark.skipif(not os.environ.get("RUN_INTEGRATION_TESTS"), reason="requires external network")
class TestFallback:
  def test_download_models_and_init_modelstate_fallback(self, tmp_path, monkeypatch):
    """The small (qcom) slot bundle loads on the hardware default device; a big (chestnut) bundle is
    bound to the AMD device, so off-hardware it must fail on the device, never load silently."""
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))
    big_bundle = ModelParser.parse_models(requests.get(ModelFetcher.MODEL_URL_CHESTNUT, timeout=30).json())[-1]
    small_bundle = ModelParser.parse_models(requests.get(ModelFetcher.MODEL_URL, timeout=30).json())[-1]

    buf = io.BytesIO()
    dump_oob(make_pkl_data(ARCHETYPES['supercombo_non20hz']), buf)
    oob_bytes = buf.getvalue()

    for bundle in (big_bundle, small_bundle):
      artifact = bundle.models[0].artifact
      for i in range(len(artifact.chunks)):
        (tmp_path / get_chunk_name(artifact.fileName, i, len(artifact.chunks))).write_bytes(oob_bytes if i == 0 else b"")

    monkeypatch.setattr(modeld_module, 'get_active_bundle', lambda params=None, *, chestnut=None: small_bundle)
    assert modeld_module.ModelState(CAM_W, CAM_H, chestnut=False).chestnut is False

    monkeypatch.setattr(modeld_module, 'get_active_bundle', lambda params=None, *, chestnut=None: big_bundle)
    with pytest.raises(Exception, match="(?i)amd|device"):
      modeld_module.ModelState(CAM_W, CAM_H, chestnut=True)
