"""Genuine GradScaler skip/recovery regression; GPU only by explicit opt-in."""
import copy
import json
import os
import unittest

import torch

from progress import gradient_fields


class Checks(unittest.TestCase):
    def test_strict_json_and_fail_closed(self):
        for norm in (0., 1.5, float('inf'), float('nan')):
            fields = gradient_fields(norm, 0)
            self.assertEqual(json.loads(json.dumps(fields, allow_nan=False)), fields)
        for norm in (float('inf'), -float('inf'), float('nan')):
            with self.assertRaises(FloatingPointError):
                gradient_fields(norm, 1)
        self.assertEqual(gradient_fields(1.5, 1)['gradient_norm'], 1.5)
        with self.assertRaises(ValueError):
            gradient_fields(1., 2)
        with self.assertRaises(ValueError):
            json.dumps({'loss': float('inf')}, allow_nan=False)

    def test_real_amp_skip_preserves_optimizer_then_recovers(self):
        device = os.environ.get('AMP_TEST_DEVICE', 'cpu')
        if device not in ('cpu', 'cuda'):
            raise ValueError('AMP_TEST_DEVICE must be cpu or cuda')
        if device == 'cpu' and not hasattr(torch.amp, 'GradScaler'):
            self.skipTest('CPU GradScaler unavailable; explicit locked CUDA regression required')
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = (torch.amp.GradScaler('cpu', init_scale=8., growth_interval=2)
                  if device == 'cpu' else torch.cuda.amp.GradScaler(init_scale=8., growth_interval=2))
        x = torch.ones(2, 2, device=device)

        def update(inject=False):
            opt.zero_grad(set_to_none=True)
            hooks = [p.register_hook(lambda g: torch.full_like(g, float('inf')))
                     for p in model.parameters()] if inject else []
            loss = model(x).square().mean()
            self.assertTrue(torch.isfinite(loss))
            scaler.scale(loss).backward()
            for h in hooks:
                h.remove()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            fields = gradient_fields(norm, int(scaler.get_scale() >= old))
            json.dumps(dict(loss=loss.item(), **fields), allow_nan=False)
            return fields

        self.assertFalse(update()['optimizer_step_skipped'])
        before = copy.deepcopy(model.state_dict())
        obefore = copy.deepcopy(opt.state_dict())
        old_scale = scaler.get_scale()
        fields = update(inject=True)
        self.assertTrue(fields['optimizer_step_skipped'])
        self.assertEqual(fields['gradient_norm_status'], 'inf')
        self.assertIsNone(fields['gradient_norm'])
        self.assertEqual(scaler.get_scale(), old_scale / 2)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        for k, state in opt.state_dict()['state'].items():
            for key, val in state.items():
                self.assertTrue(torch.equal(val, obefore['state'][k][key]))
        self.assertFalse(update()['optimizer_step_skipped'])
        self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))
        self.assertTrue(any(not torch.equal(v, before[k]) for k,v in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
