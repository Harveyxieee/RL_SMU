"""
Calibrate adaptive Sinkhorn ambiguity radii for the MMD / portfolio RDQN experiment.

Put this file at:
    Sinkhorn_RDQN/mmd/calibrate_residuals.py

Run from the Sinkhorn_RDQN repo root, for example:
    python -m mmd.calibrate_residuals \
        --data-path data/spx.csv \
        --generator-dir data/mmd_generator \
        --cal-start-date 2018-01-01 \
        --cal-end-date 2020-12-31 \
        --output-dir artifacts/calibration \
        --n-gen-samples 128 \
        --alpha 0.10 \
        --mode both

What it does:
    1. Builds rolling historical windows from real S&P 500 log prices.
    2. Uses the trained MMD generator to simulate one-step-ahead log-return samples.
    3. Computes residuals:
           abs(real_next_log_return - generated_center)
       where generated_center is either the simulated median or mean.
    4. Converts held-out residual quantiles into calibrated Sinkhorn radii:
           epsilon = scale * Quantile_{1-alpha}(residual)
    5. Saves:
           residuals.csv
           residual_calibrator.pt
           residual_calibrator.json

The resulting .pt file is intended to be loaded later by your adaptive RDQN code.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    # Works when running: python -m mmd.calibrate_residuals
    from mmd.env import GenLSTM, load_generator
except Exception:  # pragma: no cover
    # Works when running the file directly from inside mmd/
    from env import GenLSTM, load_generator


DATATYPE = torch.float32


@dataclass
class CalibrationConfig:
    data_path: str
    generator_dir: str
    output_dir: str
    cal_start_date: str
    cal_end_date: str
    state_len: int
    hist_price_len: int
    n_gen_samples: int
    batch_size: int
    alpha: float
    scale: float
    min_epsilon: float
    max_epsilon: float
    center: str
    mode: str
    vol_low_q: float
    vol_high_q: float
    device: str
    seed: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_generator_hparams(generator_path: Path) -> Dict[str, int]:
    """Infer GenLSTM constructor args from the saved state dict."""
    weights = torch.load(generator_path, map_location="cpu", weights_only=True)

    if "rnn.weight_ih_l0" not in weights or "output_net.bias" not in weights:
        raise KeyError(
            "Cannot infer generator shape. Expected keys 'rnn.weight_ih_l0' "
            "and 'output_net.bias' in generator.pt."
        )

    input_size = int(weights["rnn.weight_ih_l0"].shape[1])
    hidden_size = int(weights["rnn.weight_hh_l0"].shape[1])
    seq_dim = int(weights["output_net.bias"].shape[0])
    noise_dim = input_size - seq_dim - 1

    layer_ids = []
    for key in weights.keys():
        if key.startswith("rnn.weight_ih_l"):
            layer_ids.append(int(key.split("rnn.weight_ih_l")[-1]))
    n_lstm_layers = max(layer_ids) + 1 if layer_ids else 1

    if noise_dim <= 0:
        raise ValueError(
            f"Inferred invalid noise_dim={noise_dim}. input_size={input_size}, seq_dim={seq_dim}."
        )

    return {
        "noise_dim": noise_dim,
        "seq_dim": seq_dim,
        "hidden_size": hidden_size,
        "n_lstm_layers": n_lstm_layers,
    }


def load_mmd_generator(generator_dir: Path, device: torch.device) -> GenLSTM:
    generator_path = generator_dir / "generator.pt"
    if not generator_path.exists():
        raise FileNotFoundError(f"Missing generator checkpoint: {generator_path}")

    hp = infer_generator_hparams(generator_path)
    generator = GenLSTM(
        noise_dim=hp["noise_dim"],
        seq_dim=hp["seq_dim"],
        seq_len=1,
        hidden_size=hp["hidden_size"],
        n_lstm_layers=hp["n_lstm_layers"],
    )
    generator = load_generator(generator, str(generator_dir) + os.sep, device=device)
    generator.eval()
    return generator


def load_ma_params(generator_dir: Path) -> pd.Series:
    path = generator_dir / "ma_params.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing MA parameter file: {path}")
    params = pd.read_pickle(path)
    if "omega" not in params.index:
        raise KeyError("ma_params.pkl must contain an 'omega' parameter.")
    return params


def generate_ma_noise(
    batch_size: int,
    length: int,
    noise_dim: int,
    ma_params: pd.Series,
    device: torch.device,
) -> torch.Tensor:
    """
    Same MA-style noise generator as MMDSimulator.generate_ma_noise, but standalone.
    Output shape: (batch_size, length, noise_dim)
    """
    bias = torch.tensor(float(ma_params["omega"]), dtype=DATATYPE, device=device)
    lag_values = torch.tensor(ma_params.values[1:], dtype=DATATYPE, device=device)
    lags = lag_values.flip(0).unsqueeze(-1)  # (ma_p, 1)
    ma_p = len(lags)

    seq = torch.randn(batch_size, noise_dim, ma_p, dtype=DATATYPE, device=device)
    noise_parts = []
    for _ in range(length):
        sigma = (seq.pow(2) @ lags.expand(batch_size, -1, 1) + bias).sqrt()
        eps = sigma * torch.randn_like(sigma)
        noise_parts.append(eps)
        seq = seq.roll(-1, dims=2)
        seq[:, :, -1:] = eps

    noise = torch.cat(noise_parts, dim=2)
    return noise.permute(0, 2, 1).contiguous()


def load_price_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    required = {"Date", "log_price", "log_return", "dt"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{data_path} is missing columns: {sorted(missing)}")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_calibration_windows(
    df: pd.DataFrame,
    cal_start_date: str,
    cal_end_date: str,
    hist_price_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build windows ending at current_date and predicting target_date = next trading date.

    hist_prices[k] contains log prices from rows [start, ..., current].
    dts[k] contains dt values from rows [start + 1, ..., target], so that:
        - first hist_price_len - 1 values condition the historical transitions;
        - last value is the one-step forecast time delta.
    """
    cal_start = pd.Timestamp(cal_start_date)
    cal_end = pd.Timestamp(cal_end_date)

    hist_prices: List[np.ndarray] = []
    hist_returns_for_regime: List[np.ndarray] = []
    dts: List[np.ndarray] = []
    real_next_returns: List[float] = []
    current_dates: List[pd.Timestamp] = []
    target_dates: List[pd.Timestamp] = []

    for current_idx in range(hist_price_len - 1, len(df) - 1):
        current_date = df.loc[current_idx, "Date"]
        target_idx = current_idx + 1

        if current_date < cal_start or current_date > cal_end:
            continue

        start_idx = current_idx - hist_price_len + 1
        hist_price = df.loc[start_idx:current_idx, "log_price"].to_numpy(dtype=np.float32)
        dt_seq = df.loc[start_idx + 1:target_idx, "dt"].to_numpy(dtype=np.float32)
        real_next = float(df.loc[target_idx, "log_return"])

        # Historical returns used only for regime classification.
        hist_ret = df.loc[start_idx + 1:current_idx, "log_return"].to_numpy(dtype=np.float32)

        if (
            len(hist_price) != hist_price_len
            or len(dt_seq) != hist_price_len
            or not np.isfinite(hist_price).all()
            or not np.isfinite(dt_seq).all()
            or not np.isfinite(real_next)
            or not np.isfinite(hist_ret).all()
        ):
            continue

        hist_prices.append(hist_price)
        hist_returns_for_regime.append(hist_ret)
        dts.append(dt_seq)
        real_next_returns.append(real_next)
        current_dates.append(current_date)
        target_dates.append(df.loc[target_idx, "Date"])

    if not hist_prices:
        raise ValueError(
            "No calibration windows were built. Check cal_start_date, cal_end_date, "
            "hist_price_len, and whether dt/log_return contain NaNs."
        )

    meta_dates = np.array([[str(c.date()), str(t.date())] for c, t in zip(current_dates, target_dates)])
    return (
        np.stack(hist_prices),
        np.stack(dts),
        np.array(real_next_returns, dtype=np.float32),
        np.stack(hist_returns_for_regime),
        meta_dates,
    )


def simulate_one_step_returns(
    generator: GenLSTM,
    ma_params: pd.Series,
    hist_prices: np.ndarray,
    dts: np.ndarray,
    n_gen_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate one-step-ahead return samples for each historical window.

    Returns shape: (n_windows, n_gen_samples)
    """
    n_windows, hist_price_len = hist_prices.shape
    all_samples: List[np.ndarray] = []

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        b = end - start

        hp = torch.tensor(hist_prices[start:end], dtype=DATATYPE, device=device).unsqueeze(-1)
        dt = torch.tensor(dts[start:end], dtype=DATATYPE, device=device).unsqueeze(-1)

        hp_rep = hp.repeat_interleave(n_gen_samples, dim=0)
        dt_rep = dt.repeat_interleave(n_gen_samples, dim=0)

        noise = generate_ma_noise(
            batch_size=b * n_gen_samples,
            length=hist_price_len,
            noise_dim=generator.noise_dim,
            ma_params=ma_params,
            device=device,
        )

        with torch.no_grad():
            generated, _, _ = generator(noise=noise, dts=dt_rep, hist_x=hp_rep)

        # generated shape should be (b * n_gen_samples, 1, 1) for one-step prediction.
        generated = generated[:, -1, 0].reshape(b, n_gen_samples)
        all_samples.append(generated.detach().cpu().numpy())

    return np.concatenate(all_samples, axis=0)


def assign_regimes(
    hist_returns: np.ndarray,
    vol_low_q: float,
    vol_high_q: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simple market-regime labels based on historical volatility and trend.

    calm:      low historical volatility
    normal:    middle volatility, or high vol with non-negative trend
    high_vol:  high historical volatility with non-negative trend
    stress:    high historical volatility with negative trend
    """
    vols = np.std(hist_returns, axis=1, ddof=1)
    trends = np.mean(hist_returns, axis=1)

    low = float(np.quantile(vols, vol_low_q))
    high = float(np.quantile(vols, vol_high_q))

    regimes = np.full(len(vols), "normal", dtype=object)
    regimes[vols <= low] = "calm"
    regimes[(vols >= high) & (trends >= 0.0)] = "high_vol"
    regimes[(vols >= high) & (trends < 0.0)] = "stress"

    stats = {
        "vol_low_threshold": low,
        "vol_high_threshold": high,
        "vol_low_q": vol_low_q,
        "vol_high_q": vol_high_q,
    }
    return regimes, stats


def clipped_quantile(values: np.ndarray, q: float, scale: float, lo: float, hi: float) -> float:
    eps = float(np.quantile(values, q) * scale)
    return float(np.clip(eps, lo, hi))


def build_calibrator_artifact(
    residual_df: pd.DataFrame,
    config: CalibrationConfig,
    regime_stats: Dict[str, float],
) -> Dict:
    q = 1.0 - config.alpha
    residuals = residual_df["residual_abs"].to_numpy(dtype=float)
    global_epsilon = clipped_quantile(
        residuals,
        q=q,
        scale=config.scale,
        lo=config.min_epsilon,
        hi=config.max_epsilon,
    )

    regime_epsilons: Dict[str, float] = {}
    regime_counts: Dict[str, int] = {}
    for regime, group in residual_df.groupby("regime"):
        vals = group["residual_abs"].to_numpy(dtype=float)
        regime_counts[str(regime)] = int(len(vals))
        if len(vals) >= 20:
            regime_epsilons[str(regime)] = clipped_quantile(
                vals,
                q=q,
                scale=config.scale,
                lo=config.min_epsilon,
                hi=config.max_epsilon,
            )
        else:
            # Avoid unstable small-sample regime quantiles.
            regime_epsilons[str(regime)] = global_epsilon

    artifact = {
        "type": "residual_quantile_sinkhorn_radius",
        "description": "Calibrated epsilon values for adaptive Sinkhorn ambiguity sets.",
        "mode": config.mode,
        "alpha": config.alpha,
        "quantile_level": q,
        "global_epsilon": global_epsilon,
        "regime_epsilons": regime_epsilons,
        "regime_counts": regime_counts,
        "regime_stats": regime_stats,
        "min_epsilon": config.min_epsilon,
        "max_epsilon": config.max_epsilon,
        "scale": config.scale,
        "center": config.center,
        "state_len": config.state_len,
        "hist_price_len": config.hist_price_len,
        "n_gen_samples": config.n_gen_samples,
        "residual_summary": {
            "count": int(len(residuals)),
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals, ddof=1)),
            "median": float(np.median(residuals)),
            "q90": float(np.quantile(residuals, 0.90)),
            "q95": float(np.quantile(residuals, 0.95)),
            "q99": float(np.quantile(residuals, 0.99)),
            "max": float(np.max(residuals)),
        },
        "config": asdict(config),
    }
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate adaptive RDQN Sinkhorn epsilon from held-out residuals.")
    parser.add_argument("--data-path", type=str, default="data/spx.csv")
    parser.add_argument("--generator-dir", type=str, default="data/mmd_generator")
    parser.add_argument("--output-dir", type=str, default="artifacts/calibration")
    parser.add_argument("--cal-start-date", type=str, required=True)
    parser.add_argument("--cal-end-date", type=str, required=True)
    parser.add_argument("--state-len", type=int, default=60)
    parser.add_argument(
        "--hist-price-len",
        type=int,
        default=None,
        help="Number of log-price points used for conditioning. Default: state_len + 1, giving state_len historical returns.",
    )
    parser.add_argument("--n-gen-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.10, help="Miscoverage level; epsilon uses quantile 1-alpha.")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional multiplier applied to the residual quantile.")
    parser.add_argument("--min-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-epsilon", type=float, default=0.50)
    parser.add_argument("--center", type=str, default="median", choices=["median", "mean"])
    parser.add_argument("--mode", type=str, default="both", choices=["global", "regime", "both"])
    parser.add_argument("--vol-low-q", type=float, default=0.33)
    parser.add_argument("--vol-high-q", type=float, default=0.67)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    hist_price_len = args.hist_price_len if args.hist_price_len is not None else args.state_len + 1
    config = CalibrationConfig(
        data_path=args.data_path,
        generator_dir=args.generator_dir,
        output_dir=args.output_dir,
        cal_start_date=args.cal_start_date,
        cal_end_date=args.cal_end_date,
        state_len=args.state_len,
        hist_price_len=hist_price_len,
        n_gen_samples=args.n_gen_samples,
        batch_size=args.batch_size,
        alpha=args.alpha,
        scale=args.scale,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
        center=args.center,
        mode=args.mode,
        vol_low_q=args.vol_low_q,
        vol_high_q=args.vol_high_q,
        device=args.device,
        seed=args.seed,
    )

    if not (0.0 < config.alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if config.min_epsilon <= 0 or config.max_epsilon <= 0:
        raise ValueError("min_epsilon and max_epsilon must be positive.")
    if config.min_epsilon > config.max_epsilon:
        raise ValueError("min_epsilon cannot exceed max_epsilon.")

    device = torch.device(config.device)
    data_path = Path(config.data_path)
    generator_dir = Path(config.generator_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and generator...")
    df = load_price_data(data_path)
    generator = load_mmd_generator(generator_dir, device=device)
    ma_params = load_ma_params(generator_dir)

    print("Building calibration windows...")
    hist_prices, dts, real_next, hist_returns, meta_dates = build_calibration_windows(
        df=df,
        cal_start_date=config.cal_start_date,
        cal_end_date=config.cal_end_date,
        hist_price_len=config.hist_price_len,
    )
    print(f"Calibration samples: {len(real_next)}")

    print("Generating one-step return samples from MMD generator...")
    gen_samples = simulate_one_step_returns(
        generator=generator,
        ma_params=ma_params,
        hist_prices=hist_prices,
        dts=dts,
        n_gen_samples=config.n_gen_samples,
        batch_size=config.batch_size,
        device=device,
    )

    if config.center == "median":
        gen_center = np.median(gen_samples, axis=1)
    else:
        gen_center = np.mean(gen_samples, axis=1)

    residual_abs = np.abs(real_next - gen_center)
    regimes, regime_stats = assign_regimes(
        hist_returns=hist_returns,
        vol_low_q=config.vol_low_q,
        vol_high_q=config.vol_high_q,
    )

    residual_df = pd.DataFrame(
        {
            "current_date": meta_dates[:, 0],
            "target_date": meta_dates[:, 1],
            "real_next_log_return": real_next,
            "generated_center": gen_center,
            "generated_mean": np.mean(gen_samples, axis=1),
            "generated_median": np.median(gen_samples, axis=1),
            "generated_std": np.std(gen_samples, axis=1, ddof=1),
            "residual_abs": residual_abs,
            "hist_return_mean": np.mean(hist_returns, axis=1),
            "hist_return_vol": np.std(hist_returns, axis=1, ddof=1),
            "regime": regimes,
        }
    )

    artifact = build_calibrator_artifact(
        residual_df=residual_df,
        config=config,
        regime_stats=regime_stats,
    )

    csv_path = output_dir / "residuals.csv"
    pt_path = output_dir / "residual_calibrator.pt"
    json_path = output_dir / "residual_calibrator.json"

    residual_df.to_csv(csv_path, index=False)
    torch.save(artifact, pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("\nSaved calibration outputs:")
    print(f"  residuals:  {csv_path}")
    print(f"  torch file:  {pt_path}")
    print(f"  json file:   {json_path}")
    print("\nCalibrated epsilon summary:")
    print(f"  global_epsilon = {artifact['global_epsilon']:.8f}")
    for regime, eps in artifact["regime_epsilons"].items():
        count = artifact["regime_counts"].get(regime, 0)
        print(f"  {regime:>8s}: epsilon = {eps:.8f}, count = {count}")


if __name__ == "__main__":
    main()
