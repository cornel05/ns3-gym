# wireless-qtable baseline for ns3-gym Cognitive Radio

This repository provides a clean **Python-only** baseline for tabular Q-learning with the ns3-gym Cognitive Radio (channel selection) example.

It is intentionally focused on the channel selection setting where:
- observation is current channel occupancy,
- action is next channel choice,
- reward is +1 for no collision and -1 for collision.

## Why tabular Q-learning here?
The Cognitive Radio example often exposes low-dimensional, discrete-like observations and actions. That makes a tabular baseline simple, interpretable, and useful before trying more complex methods.

## Important scope note
- This repo is **not** a fork of ns3-gym.
- This repo contains only the Python Q-learning side.
- You must install **ns-3** and **ns3-gym** separately.

## Expected setup flow
1. Install ns-3.40.
2. Clone ns3-gym into `ns-3.40/contrib/opengym`.
3. Checkout `app-ns-3.36+` in ns3-gym.
4. Build ns-3 and install the ns3gym Python package.
5. Use this repo for tabular Q-learning training/evaluation.

See `scripts/setup_ns3gym.sh` for a documented setup template.

## Install Python dependencies
```bash
uv sync
```

Or with pip:
```bash
python -m pip install -e .
```

## Run training
```bash
python -m wireless_qtable.train --config /home/runner/work/ns3-gym/ns3-gym/configs/qlearning.yaml
```

Use `scripts/run_cognitive_radio.sh` as a placeholder reminder for running the ns-3 cognitive radio scenario from your ns-3 tree.
