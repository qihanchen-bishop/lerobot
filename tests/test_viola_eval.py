"""GUI integration checks without opening robot or camera devices."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mycode"))
from viola_eval import VIOLAEvalPolicy, config_for_checkpoint, metadata, validate_runtime
from gui_eval_lerobot_policy import EvalPolicyApp


class VIOLAEvalTests(unittest.TestCase):
    def batch(self):
        image = torch.tensor([10, 20, 200], dtype=torch.float32).reshape(1, 3, 1, 1)/255
        return {"observation.images.front": image.expand(1,3,8,12),
                "observation.images.side": image.expand(1,3,8,12),
                "observation.state": torch.arange(10,dtype=torch.float32).reshape(1,10)}

    def test_rgb_transport_and_native_state_are_preserved(self):
        payload = VIOLAEvalPolicy.observation_payload(self.batch())
        np.testing.assert_array_equal(payload['front'][0,0], [200,20,10])
        np.testing.assert_array_equal(payload['state'], np.arange(10))
        self.assertEqual(payload['side'].shape, (8,12,3))

    def test_reject_invalid_observations_before_sending(self):
        for key in ['observation.images.front','observation.state']:
            batch=self.batch();batch[key]=batch[key].clone();batch[key].flatten()[0]=float('nan')
            with self.assertRaises(ValueError):VIOLAEvalPolicy.observation_payload(batch)

    def test_native_actions_are_not_normalized_twice(self):
        policy=object.__new__(VIOLAEvalPolicy);policy.connection=Mock()
        policy._receive=Mock(return_value={'action':np.arange(10,dtype=np.float32)*10-50})
        action=policy.select_action(self.batch())
        np.testing.assert_array_equal(action.numpy(),[np.arange(10)*10-50])
        policy._receive.return_value={'action':[float('nan')]*10}
        with self.assertRaises(ValueError):policy.select_action(self.batch())

    def test_preset_discovery_joint_mapping_and_chunk_defaults(self):
        preset=ROOT/'mycode/eval_configs/finaldata/VIOLA.json'
        cfg=json.loads(preset.read_text());path=ROOT/cfg['checkpoint_path']
        self.assertIn('VIOLA',EvalPolicyApp._config_presets_in_folder(preset.parent))
        self.assertEqual(EvalPolicyApp._checkpoint_rgb_keys(path),['observation.images.front','observation.images.side'])
        ckpt,spec=metadata(path);self.assertTrue(ckpt.is_file())
        model_cfg=config_for_checkpoint(path)
        robot_names=[]
        for side in ['left','right']:
            robot_names.extend(n for n in spec['action_names'] if n.startswith(side+'_'))
            robot_names.append(side+'_gripper.pos')
        names=EvalPolicyApp._policy_feature_names({'action':{'names':robot_names}},model_cfg.output_features,'action')
        self.assertEqual(names,spec['action_names'])
        app=object.__new__(EvalPolicyApp)
        class Var:
            def __init__(self,v=''):self.v=v
            def get(self):return self.v
            def set(self,v):self.v=v
        app.vars={k:Var() for k in ['model_chunk_size','prediction_steps','n_action_steps','fusion_steps','fusion_history_weight','num_inference_steps','noise_scheduler_type','execution_mode']}
        app.use_amp=Var(True);app.enable_auto_replan=Var(True);app.lock_grippers=Var(False);app._append_log=Mock()
        app._refresh_action_chunk_settings(path)
        self.assertEqual(app.vars['n_action_steps'].get(),'1')
        self.assertEqual(app.vars['execution_mode'].get(),'synchronous')
        self.assertFalse(app.enable_auto_replan.get());self.assertTrue(app.lock_grippers.get())

    def test_incompatible_runtime_is_rejected(self):
        cfg=dict(execution_mode='synchronous',replan_interval_steps=1,prediction_steps=1,fusion_steps=0,auto_replan=False,use_amp=False,lock_grippers=True)
        with patch('viola_eval.PYTHON') as python:
            python.is_file.return_value=True
            validate_runtime(cfg)
            for key,value in [('execution_mode','asynchronous'),('replan_interval_steps',30),('auto_replan',True),('use_amp',True),('lock_grippers',False)]:
                with self.assertRaises(ValueError):validate_runtime({**cfg,key:value})

    def test_worker_timeout_and_error_are_not_silent(self):
        policy=object.__new__(VIOLAEvalPolicy);policy.connection=Mock();policy.log=SimpleNamespace(name='worker.log')
        policy.connection.poll.return_value=False
        with self.assertRaises(TimeoutError):policy._receive(.01)
        policy.connection.poll.return_value=True;policy.connection.recv.return_value={'error':'bad checkpoint'}
        with self.assertRaisesRegex(RuntimeError,'bad checkpoint'):policy._receive(.01)


if __name__=='__main__':unittest.main()
