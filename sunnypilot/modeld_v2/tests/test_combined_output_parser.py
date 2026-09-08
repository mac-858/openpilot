"""Regression for RDF's actual catalog output widths, without GPU or live IPC."""
import numpy as np
import pytest

from cereal import log
from openpilot.sunnypilot.modeld_v2.constants import Meta
from openpilot.sunnypilot.modeld_v2.fill_model_msg import PublishState, fill_model_msg, fill_pose_msg
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.parse_model_outputs_split import Parser as LegacySplitParser

# Read from RDF August 05 2026's SHA-verified catalog metadata on the device.
# Combined output has 2580 floats (including hidden state and two padding floats).
RDF_SLICES = {
  'lane_lines': slice(0, 528), 'lane_lines_prob': slice(528, 536), 'road_edges': slice(536, 800),
  'meta': slice(800, 855), 'desire_pred': slice(855, 887), 'pose': slice(887, 899),
  'wide_from_device_euler': slice(899, 905), 'road_transform': slice(905, 917),
  'plan': slice(917, 1907), 'lead': slice(1907, 2051), 'lead_prob': slice(2051, 2054),
  'desire_state': slice(2054, 2062), 'action': slice(2062, 2066), 'hidden_state': slice(2066, 2578),
}


def test_rdf_parser_through_message_serialization():
  raw = np.zeros((1, 2580), dtype=np.float32)
  plan = np.arange(495, dtype=np.float32).reshape(1, 33, 15) / 100
  lead = np.arange(72, dtype=np.float32).reshape(1, 3, 6, 4) / 10
  raw[:, 917:1412] = plan.reshape(1, -1)
  # RDF source ref a95e2c25cae5: all 72 means, then all 72 log-stds.
  raw[:, 1907:1979] = lead.reshape(1, -1)
  parsed = Parser().parse_outputs({k: raw[:, v].copy() for k, v in RDF_SLICES.items()})
  np.testing.assert_array_equal(parsed['plan'], plan)
  np.testing.assert_array_equal(parsed['lead'], lead)
  np.testing.assert_array_equal(parsed['plan_stds'], np.ones_like(plan))
  np.testing.assert_array_equal(parsed['lead_stds'], np.ones_like(lead))
  assert 'plan_weights' not in parsed and 'lead_weights' not in parsed

  # Exercise downstream schema packing, not just the reshape that originally crashed.
  driving = log.Event.new_message(drivingModelData={})
  model = log.Event.new_message(modelV2={})
  pose = log.Event.new_message(cameraOdometry={})
  action = log.ModelDataV2.Action(desiredCurvature=0, desiredAcceleration=0, shouldStop=False)
  fill_model_msg(driving, model, parsed, action, PublishState(), 1, 1, 1, 0, 123, 0.01, True, Meta)
  fill_pose_msg(pose, parsed, 1, 0, 123, True)
  assert len(model.modelV2.position.x) == 33
  assert len(model.modelV2.leadsV3) == 3
  assert all(msg.to_bytes() for msg in (driving, model, pose))


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('name,in_n,out_n,shape', [('plan', 5, 1, (33, 15)), ('lead', 2, 3, (6, 4))])
def test_legacy_hypothesis_selection_is_preserved(batch, name, in_n, out_n, shape):
  values = int(np.prod(shape))
  raw = np.zeros((batch, in_n, 2 * values + out_n), dtype=np.float32)
  for hypothesis in range(in_n):
    raw[:, hypothesis, :values] = hypothesis + 1
    raw[:, hypothesis, -out_n:] = hypothesis
  outs = {name: raw.reshape(batch, -1)}
  Parser().parse_mdn(name, outs, out_shape=shape)
  expected_shape = (batch,) + ((out_n,) if out_n > 1 else ()) + shape
  np.testing.assert_array_equal(outs[name], np.full(expected_shape, in_n))
  np.testing.assert_array_equal(outs[name + '_stds'], np.ones(expected_shape))
  assert outs[name + '_hypotheses'].shape == (batch, in_n) + shape
  np.testing.assert_allclose(outs[name + '_weights'].sum(axis=1), 1)


@pytest.mark.parametrize('name,width', [('plan', n) for n in (0, 989, 4954, 4956)] +
                         [('lead', n) for n in (0, 101, 103, 143, 145)])
def test_unknown_head_width_is_rejected(name, width):
  with pytest.raises(ValueError, match='Unsupported model output width'):
    Parser(ignore_missing=True).parse_outputs({name: np.zeros((1, width), dtype=np.float32)})


@pytest.mark.parametrize('name,width', [('plan', 1980), ('plan', 4950), ('planplus', 1980), ('lead', 48), ('lead', 96)])
def test_inferable_but_unsupported_consumer_shape_is_rejected(name, width):
  with pytest.raises(ValueError, match=f'Unsupported parsed {name} shape'):
    Parser().parse_outputs({name: np.zeros((1, width), dtype=np.float32)})


@pytest.mark.parametrize('batch', [1, 2])
def test_rdf_leads_keep_source_model_mean_and_std_order(batch):
  means = np.arange(batch * 72, dtype=np.float32).reshape(batch, 3, 24)
  log_stds = np.linspace(-2, 2, batch * 72, dtype=np.float32).reshape(batch, 3, 24)
  raw = np.concatenate((means.reshape(batch, 72), log_stds.reshape(batch, 72)), axis=-1)
  parsed = Parser().parse_outputs({'lead': raw})
  np.testing.assert_array_equal(parsed['lead'], means.reshape(batch, 3, 6, 4))
  np.testing.assert_allclose(parsed['lead_stds'], np.exp(log_stds).reshape(batch, 3, 6, 4))


def test_split_and_combined_callers_share_parser_without_double_parsing():
  assert LegacySplitParser is Parser
  parser = LegacySplitParser()
  raw = {'plan': np.arange(990, dtype=np.float32).reshape(1, -1) / 1000,
         'action': np.array([[0.1, 0.2, -1, -2]], dtype=np.float32)}
  combined = parser.parse_outputs({k: v.copy() for k, v in raw.items()})
  split = parser.parse_vision_outputs({'plan': raw['plan'].copy()})
  split.update(parser.parse_policy_outputs({'action': raw['action'].copy()}))
  for key in combined:
    np.testing.assert_array_equal(combined[key], split[key])
  np.testing.assert_array_equal(combined['action'], raw['action'][:, :2])
