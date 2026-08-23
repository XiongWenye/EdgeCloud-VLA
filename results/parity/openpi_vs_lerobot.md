# OpenPI vs LeRobot pi05 parity trace

Same fixed LIBERO Spatial observation and fixed Gaussian noise.

## Protocol

- OpenPI predicts 10 and executes 5.
- Saved LeRobot predicts 50 and executes 5.
- Matched LeRobot predicts 10 with official token IDs and executes 5.
- Both run ten Euler steps with dt=-0.1.

## Tensor differences

- Diagnostic OpenPI unrolled trace versus upstream compiled final max delta: 0.00195312.
  Final-action metrics use the upstream compiled output.
- Base image: MAE 7.785e-08, RMSE 8.942e-08, max 1.192e-07.
- Wrist image: MAE 7.695e-08, RMSE 9.315e-08, max 1.192e-07.
- Normalized state: MAE 7.223e-06, RMSE 1.763e-05, max 4.975e-05.
- Matched flow velocities: MAE 5.448e-03, RMSE 7.501e-03, max 5.768e-02.
- Matched flow states: MAE 9.846e-04, RMSE 1.460e-03, max 9.037e-03.
- Matched final normalized actions: MAE 8.257e-04, RMSE 1.225e-03, max 6.692e-03.
- Matched first five executed actions: MAE 1.508e-03, RMSE 2.131e-03, max 5.636e-03.
- Saved conversion first five executed actions: MAE 7.962e-02, RMSE 1.314e-01, max 4.953e-01.

## Semantic differences

- Valid prompt tokens: OpenPI 20, LeRobot 154; equal IDs: False.
- Official pi05_libero uses cleaned task text plus newline and does not feed state to the model.
- Saved LeRobot inserts normalized, zero-padded 32D state values discretized into 256 bins in a Task/State/Action prompt.
- OpenPI adds 1e-6 to every quantile range; LeRobot substitutes 1e-8 only for an exactly zero range.
- Saved LeRobot predicts 50 coupled tokens; OpenPI predicts 10. Both execute the first five.
