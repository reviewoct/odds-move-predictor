import numpy as np
import torch

from oddsnet import OddsMoveNet, SimConfig, WindowSampler, build_features, make_labels, simulate
from oddsnet.features import DOWN, FLAT, UP


def small_sim(n=3000, seed=1):
    return simulate(SimConfig(n_steps=n, seed=seed))


def test_simulator_shapes_and_determinism():
    a, b = small_sim(), small_sim()
    assert a["prices"].shape == (3000, 5)
    assert np.array_equal(a["prices"], b["prices"])
    assert (a["prices"] > 1.0).all()
    assert set(np.unique(a["regime"])) <= {0, 1, 2}


def test_prices_sit_on_tick_grid():
    p = small_sim()["prices"].astype(np.float64)
    assert np.allclose(p * 100, np.round(p * 100), atol=1e-3)


def test_labels_match_bruteforce():
    prices = small_sim(500)["prices"]
    H = 5
    labels = make_labels(prices, H)
    logp = np.log(prices.astype(np.float64))
    for t in (10, 123, 400):
        for k in range(prices.shape[1]):
            expected = FLAT
            for h in range(1, H + 1):
                d = logp[t + h, k] - logp[t + h - 1, k]
                if d != 0:
                    expected = UP if d > 0 else DOWN
                    break
            assert labels[t, k] == expected


def test_features_are_causal():
    """Changing the future must not change features at or before time t."""
    prices = small_sim(800)["prices"]
    mean = np.zeros(prices.shape[1])
    f1, _ = build_features(prices, spread_mean=mean)
    tampered = prices.copy()
    tampered[500:] *= 1.5
    f2, _ = build_features(tampered, spread_mean=mean)
    assert np.array_equal(f1[:500], f2[:500])


def test_sampler_window_alignment():
    prices = small_sim(400)["prices"]
    feats, _ = build_features(prices)
    labels = make_labels(prices, 5)
    s = WindowSampler(feats, labels, window=16, horizon=5)
    lo, hi = s.valid_range(0, 400)
    assert (lo, hi) == (15, 395)
    x, y = s.batch(torch.tensor([lo, 100]))
    assert x.shape == (2, 16, 5, 3) and y.shape == (2, 5)
    assert torch.equal(x[1, -1], torch.from_numpy(feats[100]))
    assert torch.equal(x[1, 0], torch.from_numpy(feats[85]))


def test_model_shapes_and_probabilities():
    model = OddsMoveNet(n_books=5)
    x = torch.randn(4, 32, 5, 3)
    assert model(x).shape == (4, 5, 3)
    p = model.predict_proba(x)
    assert torch.allclose(p.sum(-1), torch.ones(4, 5), atol=1e-5)
    _, attn = model.encode(x)
    assert attn.shape == (4, 5, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(4, 5), atol=1e-5)


def test_temperature_scaling_reduces_nll_of_overconfident_logits():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (2000,))
    logits = torch.randn(2000, 3)
    logits[torch.arange(2000), labels] += 1.0
    logits *= 5.0  # make them badly over-confident
    model = OddsMoveNet(n_books=5)
    before = torch.nn.functional.cross_entropy(logits, labels)
    t = model.fit_temperature(logits, labels)
    after = torch.nn.functional.cross_entropy(logits / t, labels)
    assert t > 1.0 and after < before


def test_model_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    prices = small_sim(600)["prices"]
    feats, _ = build_features(prices)
    labels = make_labels(prices, 10)
    s = WindowSampler(feats, labels, 32, 10)
    x, y = s.batch(torch.arange(100, 132))
    model = OddsMoveNet(5)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = None
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
