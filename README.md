# OmniXtreme

[Author project](https://extreme-humanoid.github.io/) · [Checkpoints](https://github.com/Perkins729/OmniXtreme/tree/6f2716d1b1d1132d513d86beb2193f5b18f11fcd/policy) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-omnixtreme.git
cd MotionChallenge-omnixtreme
uv sync
uv run evaluate.py
```

Produces success, completion and six MPBPE values in a verified `result.json`.
MuJoCo 3.17 CPU batch; `--jobs` sets parallelism. Missing data and the three
author weights are downloaded and SHA256-checked. Results and traces stay outside Git.

Use local files with `--weights BASE --artifact fk=FK --artifact residual=RESIDUAL
--data DATA --offline`. Policy sampling uses the recorded per-clip seed.
The root `MotionChallenge-Core` submodule owns data, physics, scoring and verification.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/11873584f838ba29fd2a2327ca15016192c3f3e1/docs/OBSERVATIONS.md).
