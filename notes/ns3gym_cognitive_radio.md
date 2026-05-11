# ns3-gym cognitive radio baseline notes

## Mapping used in this baseline
- **State**: channel occupation state in current time slot (discretized tuple).
- **Action**: channel selection for the next time slot.
- **Reward**: +1 no collision, -1 collision.
- **Done/Gameover**: more than 3 collisions in the last 10 time-slots.

## Why tabular Q-learning is feasible
For small channel counts and occupancy vectors, state/action spaces stay compact enough for a Q-table and provide a clear baseline before deep RL.

## What can break in practice
- ns-3 / ns3-gym version mismatch.
- Wrong branch (`app-ns-3.36+`) or incorrect clone path (`ns-3.40/contrib/opengym`).
- Observation format may differ from assumptions (binary vs continuous).
- `info` dict keys for collision metrics may vary by scenario implementation.

## Next steps after baseline works
1. Confirm exact observation and action-space definitions from local scenario.
2. Tune epsilon schedule and episode length.
3. Add simple logging and plotting for training curves.
4. Compare against random policy and fixed-channel baselines.
