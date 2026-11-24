# odds-move-predictor

A small PyTorch proof of concept for predicting the direction of the next
price move across several bookmakers, treating each bookmaker's pricing
engine as a black box.

> **Scope.** This is a prototype on **synthetic** data. The numbers below show
> that the pipeline recovers structure that is known to be in the simulator;
> they are not a claim about performance on real bookmaker feeds.

## What it does

| Piece | File | Notes |
|---|---|---|
| Market simulator | `oddsnet/simulate.py` | 5 bookmakers tracking a latent fair price, each with its own reaction speed, dead-band, margin and tick grid. A hidden Markov chain switches calm / active / shock regimes. |
| Features + labels | `oddsnet/features.py` | Causal features only (last return, de-meaned deviation from cross-book consensus, time since last move). Label = direction of each book's next move within a horizon. |
| Model | `oddsnet/model.py` | Shared GRU encoder per book + learned book embedding, then multi-head attention across books as a dynamic "who leads whom" graph, then a 3-way head (down / flat / up). ~51k parameters. |
| Calibration | `OddsMoveNet.fit_temperature` | Temperature scaling fitted on the validation split. |
| Training / evaluation | `train.py` | Chronological 70 / 15 / 15 split with a horizon-sized gap at each boundary so labels never look across it. |
| Latency benchmark | `benchmark.py` | Single-thread CPU inference timings. |
| Tests | `tests/` | Simulator determinism, label correctness vs brute force, feature causality (no look-ahead), window alignment, model shapes, calibration, overfit sanity check. |

## Results (seed 0, 200k simulated ticks, test split = last 15 %)

Horizon = 10 ticks, window = 32 ticks. Laptop CPU, 3 epochs, ~105 s training.

| Metric | Value |
|---|---|
| Directional hit rate (samples where the price moved) | **75.1 %** |
| — baseline: follow the consensus gap | 68.3 % |
| — baseline: momentum (repeat last move) | 49.1 % |
| 3-class accuracy (down / flat / up) | 77.2 % |
| — baseline: always predict "flat" | 75.1 % |
| Expected calibration error | 0.012 |

Directional hit rate per bookmaker, fastest to slowest:
64.6 %, 75.4 %, 83.0 %, 89.0 %, 95.2 %. The slower a book reacts, the more
predictable its next move is from the others, which is what the simulator is
built to produce.

The learned attention weights recover the leader without being told: every
follower book puts 40-45 % of its attention on book 0 (the fastest one),
against 20 % if attention were uniform.

Temperature scaling made no measurable difference here (T = 1.02; ECE 0.012
before and after): the model was already well calibrated on this data. It is
kept because real feeds are non-stationary and usually do need it.

### Latency (single CPU thread, PyTorch eager mode)

| Batch | p50 | p99 | Market snapshots / s |
|---|---|---|---|
| 1 | 0.54 ms | 2.0 ms | ~1,500 |
| 64 | 7.5 ms | 19 ms | ~8,000 |
| 512 | 61 ms | 75 ms | ~8,300 |

One "snapshot" is a full 32-tick window over all 5 books and yields
predictions for all 5. These are model-only timings: no gRPC, no feature
computation, no network.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python train.py          # writes artifacts/model.pt and artifacts/results.json
python benchmark.py      # writes artifacts/latency.json
```

Or with Docker (runs tests, then training and the benchmark):

```bash
docker build -t odds-move-predictor .
docker run --rm odds-move-predictor
```

## What is not here yet

- Real tick data. Real feeds add irregular timestamps, suspensions, missing
  quotes and many markets per event; none of that is modelled.
- A gRPC serving layer and streaming feature state.
- Explicit regime inference. The GRU state carries regime information
  implicitly; there is no separate HMM / switching state-space model, and no
  test of how well the hidden regime can be read out.
- Online adaptation and drift monitoring.
- TorchScript / ONNX export or quantisation for lower latency.
