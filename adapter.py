"""OmniXtreme's released flow policy, residual policy and FK network on CPU."""
from motion_challenge.adapters.onnx import cpu_session
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from motion_challenge.adapters.onnx import ONNXController
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file
from native_rotations import quat_conjugate, quat_mul, my_quaternion_to_matrix


class OmniXtreme(ONNXController):
    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        torch.set_num_threads(1)
        self.perm = self.cfg['PERM']
        self.inv_perm = self.cfg['INV_PERM']
        self.default_t = torch.tensor(self.default, dtype=torch.float32)
        self.scale_t = torch.tensor(self.scale, dtype=torch.float32)
        self.kp_t = torch.tensor(self.kp, dtype=torch.float32)
        self.kd_t = torch.tensor(self.kd, dtype=torch.float32)
        self.envelope = {key: torch.tensor(value, dtype=torch.float32)
                         for key, value in self.cfg['envelope'].items()}
        self.extra = {}
        for name in ('fk', 'residual'):
            declaration = runtime['artifacts'][name]
            path = root/declaration['path']
            if sha256_file(path) != declaration['sha256']:
                raise ValueError(f'{name} checkpoint checksum mismatch')
            self.extra[name] = cpu_session(path, declaration['sha256'])

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.history = torch.zeros((15, 90), dtype=torch.float32)
        self.previous_base = torch.zeros(29, dtype=torch.float32)
        self.previous_combined = torch.zeros(29, dtype=torch.float32)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.initial_reference_rotation = R.from_quat(np.asarray(reference[0].root_orientation_xyzw, np.float32))
        self.initial_yaw = 0.
        self.first = True

    def observations(self, state, reference, initial_noise=None):
        q, qd, omega, _, _ = self.proprio(state)
        self.q = torch.from_numpy(q)
        self.qd = torch.from_numpy(qd)
        angular = torch.from_numpy(omega)
        rpy = R.from_quat(np.asarray(state.root_orientation_xyzw, np.float32)).as_euler('xyz')
        yaw = rpy[2]-self.initial_yaw
        while yaw > np.pi:
            yaw -= 2*np.pi
        while yaw < -np.pi:
            yaw += 2*np.pi
        body_rotation = R.from_euler('xyz', [rpy[0], rpy[1], yaw])
        if self.first:
            self.initial_yaw = -(self.initial_reference_rotation*body_rotation.inv()).as_euler('xyz')[2]
            self.first = False
        # This is the author's FK network; only its torso rotation is used.
        _, torso_wxyz = self.extra['fk'].run(None, {
            'joint_angles': q[12:15][None].astype(np.float32),
            'base_pos': np.asarray([[0., 0., .79]], np.float32),
            'base_quat': body_rotation.as_quat()[[3, 0, 1, 2]][None].astype(np.float32),
        })
        target = reference[1]
        torso = next(b for b in target.bodies if b.name == 'torso_link')
        actual_q = torch.from_numpy(torso_wxyz[:, [1, 2, 3, 0]])
        target_q = torch.tensor([torso.orientation_xyzw], dtype=torch.float32)
        relative = quat_mul(quat_conjugate(actual_q, True), target_q, True)
        orientation = my_quaternion_to_matrix(relative, True)[..., :2].reshape(6)
        ref_q = torch.tensor(np.asarray(target.joint_position_rad)[self.order], dtype=torch.float32)
        ref_qd = torch.tensor(np.asarray(target.joint_velocity_rad_s)[self.order], dtype=torch.float32)
        qrel = self.q-self.default_t
        real = torch.cat([qrel, self.qd, angular, self.previous_base])
        self.history[:-1] = self.history[1:].clone()
        self.history[-1] = real
        command = torch.cat([ref_q, ref_qd, orientation])
        residual_real = torch.cat([angular, qrel[self.inv_perm], self.qd[self.inv_perm],
                                   self.previous_combined[self.inv_perm]])
        residual_command = torch.cat([ref_q[self.inv_perm], ref_qd[self.inv_perm], orientation])
        self.residual_prefix = torch.cat([residual_command, residual_real]).float().numpy()
        noise = (torch.randn((1, 29), generator=self.generator, dtype=torch.float32).numpy()
                 if initial_noise is None else np.asarray(initial_noise, dtype=np.float32))
        return {'real_obs': real.float().numpy()[None], 'command_obs': command.numpy()[None],
                'real_historical_obs_raw': self.history.numpy().reshape(1, -1), 'initial_noise': noise}

    def targets(self, combined):
        desired = combined*self.scale_t+self.default_t
        x1, x2, y1, y2 = (self.envelope[k] for k in ('X1', 'X2', 'Y1', 'Y2'))
        absolute = self.qd.abs()
        over = torch.clamp(absolute-x1, min=0.)
        positive = torch.where(absolute <= .01, y2, torch.where(self.qd >= 0., y1, y2))
        negative = torch.where(absolute <= .01, -y2, torch.where(self.qd >= 0., -y2, -y1))
        high = torch.clamp(positive-positive/torch.clamp(x2-x1, min=1e-6)*over, min=0.)
        low = torch.clamp(negative+(-negative)/torch.clamp(x2-x1, min=1e-6)*over, max=0.)
        return torch.clip(desired, (low+self.kd_t*self.qd)/self.kp_t+self.q,
                          (high+self.kd_t*self.qd)/self.kp_t+self.q).numpy()

    def act(self, state, reference):
        inputs = self.observations(state, reference)
        base = self.infer(inputs)['action'].reshape(29)
        residual_input = np.concatenate([self.residual_prefix, base[self.inv_perm]])[None]
        residual = self.extra['residual'].run(None, {'obs': residual_input})[0].reshape(29)
        combined = base + np.float32(self.cfg['residual_gain'])*residual[self.perm]
        if not np.isfinite(combined).all():
            raise ValueError('nonfinite OmniXtreme action')
        self.previous_base = torch.from_numpy(base.copy())
        self.previous_combined = torch.from_numpy(combined.copy())
        self.last = combined.copy()
        target = self.targets(self.previous_combined)
        def ordered(array):
            return tuple(float(x) for x in array[self.inverse])
        return ControlCommand(ordered(target), (0.,)*29, ordered(self.kp), ordered(self.kd), (0.,)*29)

    def close(self):
        self.extra.clear()
        super().close()


def create(root, config):
    return OmniXtreme(root, config)
