"""Concurrent native inference with independent per-motion state."""
from motion_challenge.adapters.parallel import ParallelControllers
from restricted import create as create_single


def create(root, runtime):
    return ParallelControllers(root, runtime, create_single)
