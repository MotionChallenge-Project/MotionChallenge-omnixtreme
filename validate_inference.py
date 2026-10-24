"""Compare nominal-state transport and three-network inference to author code."""
import argparse
import ast
import contextlib
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
import torch

from adapter import OmniXtreme
from motion_challenge.physics import World
from motion_challenge.sdk import ReferenceFrame
from motion_challenge.validation import external_output_path


def load_author(source):
    def options():
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        return o
    namespace = {'np': np, 'torch': torch, 'R': R, 'os': os,
                 'ort': NS(SessionOptions=options, InferenceSession=ort.InferenceSession,
                           GraphOptimizationLevel=ort.GraphOptimizationLevel,
                           get_available_providers=ort.get_available_providers)}
    def execute(nodes, path):
        exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+nodes, []),
                     str(path), 'exec'), namespace)
    rotation_path = source/'isaac_utils/rotations.py'
    functions = [n for n in ast.parse(rotation_path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in
                 ('quat_conjugate', 'quat_mul', 'my_quaternion_to_matrix', 'xyzw_to_wxyz')]
    for n in functions:
        n.decorator_list = []
    execute(functions, rotation_path)
    namespace['_rot'] = NS(**{n.name: namespace[n.name] for n in functions})
    wrapper_path = source/'residual_policy.py'
    execute([n for n in ast.parse(wrapper_path.read_text()).body if isinstance(n, ast.ClassDef)], wrapper_path)
    deployment = source/'deploy_mujoco.py'
    tree = ast.parse(deployment.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) or
             isinstance(n, ast.FunctionDef) and n.name in
             ('normalize_angle', 'get_euler_xyz', 'rot_subtract_frame_transforms')]
    nodes.append(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeployNode')
    loop = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'main_loop')
    while_node = next(n for n in loop.body if isinstance(n, ast.While))
    body = next(n for n in while_node.body if isinstance(n, ast.If)).body
    stop = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                and any(ast.unparse(t) == 'self.angles' for t in n.targets))
    step = ast.FunctionDef(name='native_step', args=loop.args, body=body[:stop+1], decorator_list=[],
                          returns=None, type_comment=None, type_params=[])
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('lowlevel_state_mujoco', 'compute_observations')]+[step]
    nodes.append(ast.fix_missing_locations(node))
    execute(nodes, deployment)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    pins = json.loads((root/'native-sources.json').read_text())
    for file, digest in pins['files'].items():
        assert hashlib.sha256((args.source/file).read_bytes()).hexdigest() == digest
    native = load_author(args.source)
    runtime = json.loads((root/'runtime.json').read_text())
    policy = OmniXtreme(root, runtime)
    original_observations = policy.observations
    captured = {}
    supplied_noise = None
    def capture(state, reference):
        obs = original_observations(state, reference, initial_noise=supplied_noise)
        captured.update({k: v.copy() for k, v in obs.items()})
        return obs
    policy.observations = capture
    world = World(root/'MotionChallenge-Core/robots/g1-29dof-cpu-v0.json')
    assert list(world.joint_names) == policy.names
    native_env = native['G1']()
    np.testing.assert_array_equal(policy.kp, native_env.p_gains)
    np.testing.assert_array_equal(policy.kd, native_env.d_gains)
    np.testing.assert_array_equal(policy.scale_t.numpy(), native_env.action_scale.numpy()[0])
    for name in ('X1', 'X2', 'Y1', 'Y2'):
        np.testing.assert_array_equal(policy.envelope[name], getattr(native_env, name))
    base = native['OnnxBasePolicyWrapper'](str(root/runtime['weights']), device='cpu')
    residual = native['OnnxResidualPolicyWrapper'](str(root/runtime['artifacts']['residual']['path']), device='cpu')
    rng = np.random.default_rng(241024)
    permutation = rng.permutation(29)
    shuffled_names = tuple(np.asarray(world.joint_names)[permutation])
    canonical = [shuffled_names.index(n) for n in world.joint_names]
    def shuffled(packet):
        kw = dict(joint_position_rad=tuple(np.asarray(packet.joint_position_rad)[permutation]),
                  joint_velocity_rad_s=tuple(np.asarray(packet.joint_velocity_rad_s)[permutation]))
        if hasattr(packet, 'joint_names'):
            kw['joint_names'] = shuffled_names
        return replace(packet, **kw)
    errors = {k: 0. for k in ('real_obs', 'command_obs', 'history', 'residual_obs', 'base_action',
                              'combined_action', 'motor_target')}
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    clipped_cases = 0
    for reset in range(4):
        refs, qposs, qvels = [], [], []
        for step in range(40):
            rotation = R.from_euler('xyz', rng.uniform(-.8, .8, 3))
            world.data.qpos[:] = np.r_[rng.normal(0., .3, 2), .8,
                rotation.as_quat()[[3, 0, 1, 2]], rng.normal(0., .2, 29)]
            world.data.qvel[:] = rng.normal(0., .5, 35)
            world.data.time = step*.02
            mujoco.mj_forward(world.model, world.data)
            state = world.state()
            refs.append(ReferenceFrame(step*.02, step*.02, state.joint_position_rad,
                state.root_position_world_m, state.root_orientation_xyzw, state.joint_velocity_rad_s,
                state.root_linear_velocity_world_m_s, state.root_angular_velocity_world_rad_s, state.bodies))
            qposs.append(world.data.qpos.copy())
            world.data.qvel[6:] *= 60. if step % 4 == 0 else 1.
            qvels.append(world.data.qvel.copy())
        refs = refs[:35]
        n = native['DeployNode'].__new__(native['DeployNode'])
        n.device = 'cpu'
        n.env = native_env
        n.env.mj_data = NS(qpos=qposs[0], qvel=qvels[0])
        n.base_policy, n.residual_policy = base, residual
        n.fk_quat_session = policy.extra['fk']
        n.clip_mode, n.residual_gain = 'envelope', 1.
        n.config = NS(noise_scales=NS(**{k: 0. for k in ('base_ang_vel', 'base_euler_xyz', 'dof_vel',
                                'ref_joint_angles', 'ref_joint_velocities', 'dof_pos')}),
                      obs_scales=NS(**{k: 1. for k in ('base_ang_vel', 'base_euler_xyz', 'dof_vel',
                                'ref_joint_angles', 'ref_joint_velocities', 'dof_pos')}))
        n.prev_action = torch.zeros(29)
        n.prev_prev_action = torch.zeros(29)
        n.prev_base_action = torch.zeros(29)
        n._last_base_action_env = torch.zeros((1, 29))
        n.history_data = torch.zeros((15, 90))
        n.obs_dict = {'real_obs': torch.zeros((1, 90)), 'command_obs': torch.zeros((1, 64)),
                      'residual_obs': torch.zeros((1, 183)), 'real_historical_obs_raw': n.history_data[None]}
        n.episode_length_buf = 0
        n.initial_yaw = n.init_motion_yaw = 0.
        n.start_policy = n.update_initial_yaw = True
        n.init_motion_rot_quat = torch.tensor(refs[0].root_orientation_xyzw, dtype=torch.float32)
        n.quat_xyzw_r = np.asarray([0., 0., 0., 1.])
        n.quat_wxyz_r = np.asarray([1., 0., 0., 0.])
        n.bm_anchor_id = 0
        for step in range(40):
            ref = refs[min(step+1, len(refs)-1)]
            torso = next(b for b in ref.bodies if b.name == 'torso_link')
            frame = dict(rg_pos_t=torch.tensor([[torso.position_world_m]], dtype=torch.float32),
                         rg_rot_t=torch.tensor([[torso.orientation_xyzw]], dtype=torch.float32),
                         dof_pos=torch.tensor([ref.joint_position_rad], dtype=torch.float32),
                         dof_vel=torch.tensor([ref.joint_velocity_rad_s], dtype=torch.float32))
            n.motion_res_buf = [frame, frame]
            n.episode_length_buf = 0
            n.env.mj_data = NS(qpos=qposs[step], qvel=qvels[step])
            world.data.qpos[:], world.data.qvel[:] = qposs[step], qvels[step]
            world.data.time = step*.02
            mujoco.mj_forward(world.model, world.data)
            state = shuffled(world.state())
            window = (shuffled(refs[min(step, len(refs)-1)]), shuffled(ref))
            if step == 0:
                policy.reset(state, window, reset)
            with contextlib.redirect_stdout(io.StringIO()):
                n.native_step()
            supplied_noise = n.obs_dict['initial_noise'].numpy()
            control = policy.act(state, window)
            compare('real_obs', captured['real_obs'], n.obs_dict['real_obs'])
            compare('command_obs', captured['command_obs'], n.obs_dict['command_obs'])
            compare('history', captured['real_historical_obs_raw'], n.history_data.reshape(1, -1))
            compare('residual_obs', policy.residual_prefix, n.obs_dict['residual_obs'].numpy()[0, :-29])
            compare('base_action', policy.previous_base, n.prev_base_action)
            compare('combined_action', policy.previous_combined, n.prev_action)
            compare('motor_target', np.asarray(control.joint_position_target_rad)[canonical], n.angles)
            desired = (n.prev_action*n.env.action_scale+n.env.default_dof_pos).numpy().reshape(-1)
            clipped_cases += not np.array_equal(desired, n.angles)
    policy.observations = original_observations
    infer = policy.infer
    noise_samples = []
    def capture_noise(inputs):
        noise_samples.append(inputs['initial_noise'].copy())
        return infer(inputs)
    policy.infer = capture_noise
    sequences = []
    for seed in (2026, 2026, 2027):
        policy.reset(state, window, seed)
        sequences.append([policy.act(state, window).joint_position_target_rad for _ in range(4)])
    np.testing.assert_array_equal(sequences[0], sequences[1])
    np.testing.assert_array_equal(noise_samples[:4], noise_samples[4:8])
    expected = torch.randn((1, 29), generator=torch.Generator().manual_seed(2026)).numpy()
    np.testing.assert_array_equal(noise_samples[0], expected)
    assert not np.array_equal(sequences[0], sequences[2])
    assert not np.array_equal(noise_samples[:4], noise_samples[8:])
    policy.close()
    report = {'status': 'passed' if max(errors.values()) < 2e-5 else 'failed', 'source': pins,
        'cases': 160, 'resets': 4, 'endpoint_cases': 24, 'envelope_clipped_cases': clipped_cases,
        'shuffled_joint_order': True, 'max_abs_errors': errors, 'tolerance': 2e-5,
        'seeded_sampling': {'same_seed_reset_exact': True, 'different_seed_changes_output': True,
                           'matches_torch_standard_normal': True, 'steps_per_sequence': 4},
        'noise': 'Same author-generated initial_noise supplied to both networks; sensor noise zero for nominal protocol.',
        'physics': 'Author per-physics-step friction and simulator settings are excluded; Core owns common physics.'}
    output = external_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    assert report['status'] == 'passed'


if __name__ == '__main__':
    main()
