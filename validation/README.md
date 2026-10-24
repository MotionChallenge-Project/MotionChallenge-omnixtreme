# Author inference validation

The adapter uses the original flow base, residual and FK ONNX files from
`Perkins729/OmniXtreme` at `6f2716d1b1d1132d513d86beb2193f5b18f11fcd`.
The three hashes and download locations are recorded in `provenance.json`.

```bash
git clone https://github.com/Perkins729/OmniXtreme.git /tmp/OmniXtreme
git -C /tmp/OmniXtreme checkout 6f2716d1b1d1132d513d86beb2193f5b18f11fcd
uv run validate_inference.py --source /tmp/OmniXtreme \
  --output /tmp/omnixtreme-native-parity.json
```

This optional audit expects all three weights in `checkpoints/`, named as in
`runtime.json`. It extracts the original observation functions, ONNX wrappers,
quaternion operations and the inference/target section of the author's main loop.
Every source file is SHA256-checked first; no viewer or author physics is run.

Across 160 states, four resets and 24 endpoint clamps, all policy observations,
histories, base actions, residual combinations and final motor targets match
exactly. The states include shuffled joint ordering and high joint velocities;
126 cases exercise the native torque-speed target envelope. Two resets with
the same seed produce identical outputs; a different seed changes them.

The base network's standard-normal `initial_noise` remains stochastic. Each
clip uses a dedicated seeded CPU PyTorch generator. Conditional inference is
compared using exactly the same noise tensor on both sides. The author's unused
zero-amplitude sensor random draws and random warm-up do not enter that generator.

The nominal benchmark disables sensor noise, including the author's configured
angular-velocity noise of 0.1. Its extra motor-friction simulation, robot XML,
fixed initial height and physics stepping are replaced by the common Core rules.
The native control-tick target envelope, gains, action scales, reference timing,
history, FK, residual gain and three checkpoint networks are retained.
