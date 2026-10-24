"""Restricted-track bridge: local FK only, never measured translation."""
from adapter import create as create_native
from motion_challenge.adapters.kinematics import KinematicStateView
from motion_challenge.resources import lafan_paths


class RestrictedController:
    def __init__(self, root, runtime):
        self.inner = create_native(root, runtime)
        self.kinematics = KinematicStateView(lafan_paths()['robot'])

    def reset(self, state, reference, seed):
        self.inner.reset(self.kinematics.reconstruct(state), reference, seed)

    def act(self, state, reference):
        return self.inner.act(self.kinematics.reconstruct(state), reference)

    def close(self):
        self.inner.close()


def create(root, runtime):
    return RestrictedController(root, runtime)
