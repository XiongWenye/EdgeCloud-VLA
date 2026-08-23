# OpenPI vs Aligned LeRobot pi05 Parity Trace

Same fixed LIBERO Spatial observation and fixed Gaussian noise.

## Protocol

- OpenPI predicts 10 and executes 5.
- Aligned LeRobot predicts 10 and executes 5.
- Both run 10 Euler flow-matching steps with dt=-0.1.

## Tensor Parity Metrics

- Base image: MAE 7.785e-08, RMSE 8.942e-08, max 1.192e-07.
- Wrist image: MAE 7.695e-08, RMSE 9.315e-08, max 1.192e-07.
- Camera masks: OpenPI [True, True, False], LeRobot [True, True, False].
- Aligned flow velocities: MAE 5.617e-03, RMSE 7.742e-03, max 5.768e-02.
- Aligned flow states: MAE 9.890e-04, RMSE 1.442e-03, max 9.037e-03.
- Aligned final normalized actions: MAE 8.538e-04, RMSE 1.255e-03, max 5.716e-03.
- Aligned first five executed actions: MAE 1.477e-03, RMSE 2.023e-03, max 4.814e-03.
- Saved unaligned conversion first five executed actions: MAE 7.963e-02, RMSE 1.316e-01, max 4.961e-01.

## Token and Semantic Alignment

- Valid prompt tokens: OpenPI 20, Aligned LeRobot 20; equal IDs: True.
- Official pi05_libero uses cleaned task text plus newline and does not feed state to the model.
- Aligned LeRobot preprocessor independently produces the exact same token IDs and masks using local PaliGemma SentencePiece.
- OpenPI quantile normalization formula with epsilon 1e-6 is matched exactly in LeRobot normalizer/unnormalizer.
- Aligned LeRobot predicts 10 coupled tokens and executes 5, matching OpenPI replanning horizon.
