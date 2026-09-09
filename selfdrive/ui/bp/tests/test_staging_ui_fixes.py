from types import SimpleNamespace

import pytest

from openpilot.selfdrive.ui.sunnypilot.layouts.settings import models
from openpilot.system.ui.widgets import label


def test_cache_size_tolerates_disappearing_file(tmp_path, monkeypatch):
  monkeypatch.setattr(models, 'CUSTOM_MODEL_PATH', str(tmp_path))
  monkeypatch.setattr(models.os, 'listdir', lambda _: ['present', 'removed'])
  seen = []

  def size(path):
    seen.append(path)
    if path.endswith('removed'):
      raise FileNotFoundError(path)
    return 2048

  monkeypatch.setattr(models.os.path, 'getsize', size)
  assert models.ModelsLayout.calculate_cache_size() == 2048 / 1024**2
  assert len(seen) == 2


@pytest.mark.parametrize('fps', [10, 20, 30, 60, 120])
def test_scroll_distance_is_independent_of_target_fps(fps, monkeypatch):
  monkeypatch.setattr(label, 'gui_app', SimpleNamespace(target_fps=fps))
  for name in ('begin_scissor_mode', 'end_scissor_mode', 'draw_rectangle_gradient_h'):
    monkeypatch.setattr(label.rl, name, lambda *args: None)
  widget = SimpleNamespace(
    _rect=SimpleNamespace(x=0, y=0, width=100, height=40), _max_width=None,
    _update_text_cache=lambda *args: None, _cached_wrapped_lines=['long text'],
    _cached_line_sizes=[SimpleNamespace(x=1000, y=20)], _cached_line_emojis=[[]],
    _line_height=1, _elide=False, _alignment_vertical=label.rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
    _needs_scroll=True, _font_size=20, _scroll_state=label.ScrollState.SCROLLING,
    _scroll_offset=0, _render_line=lambda *args: None,
  )
  for _ in range(fps):
    label.UnifiedLabel._render(widget, None)
  assert widget._scroll_offset == pytest.approx(-48)
