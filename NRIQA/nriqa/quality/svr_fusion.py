from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.svm import SVR

from nriqa.utils.metrics import plcc, rmse, srcc


def load_feature(base_dir: str | Path, split: str, layer_name: str):
    return np.load(Path(base_dir) / split / f"{layer_name}.npy")


def load_scores(base_dir: str | Path, split: str):
    return np.load(Path(base_dir) / split / "scores.npy")


def load_layer_names(base_dir: str | Path):
    with open(Path(base_dir) / "train" / "layer_names.json", "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_feature(x):
    return x.reshape(x.shape[0], -1)


def minmax_norm_with_train_stats(X_train, X_test):
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0
    return (X_train - train_min) / denom, (X_test - train_min) / denom


def preselect_by_std(X_train, X_test, k_value=100000, ascending=True):
    stds = np.std(X_train, axis=0)
    sorted_idx = np.argsort(stds)
    if not ascending:
        sorted_idx = sorted_idx[::-1]

    selected_idx = sorted_idx[: min(k_value, X_train.shape[1])]
    return X_train[:, selected_idx], X_test[:, selected_idx], selected_idx


def fast_pca_like_matlab(X_train_centered, X_test_centered, n_components):
    X = X_train_centered
    n_samples, n_features = X.shape
    small_cov = (X @ X.T) / max(n_features - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(small_cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    keep = eigvals > 1e-10
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    n_components = min(n_components, len(eigvals))
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    coeff_small = eigvecs / np.sqrt(np.maximum((n_features - 1) * eigvals, 1e-12))
    coeff = X.T @ coeff_small

    X_train_pca = X @ coeff
    X_test_pca = X_test_centered @ coeff
    explained = eigvals / np.maximum(eigvals.sum(), 1e-12)
    return X_train_pca, X_test_pca, explained.sum()


def run_single_layer_svr(
    X_train,
    X_test,
    y_train,
    k_value=100000,
    n_components=75,
    preselect_ascending=True,
    C=25,
    gamma="scale",
    epsilon=0.1,
):
    X_train = flatten_feature(X_train)
    X_test = flatten_feature(X_test)
    original_dim = X_train.shape[1]

    X_train_sel, X_test_sel, _ = preselect_by_std(
        X_train, X_test, k_value=k_value, ascending=preselect_ascending
    )
    selected_dim = X_train_sel.shape[1]

    X_train_norm, X_test_norm = minmax_norm_with_train_stats(X_train_sel, X_test_sel)
    train_mean = X_train_norm.mean(axis=0)
    X_train_centered = X_train_norm - train_mean
    X_test_centered = X_test_norm - train_mean

    n_components = min(n_components, X_train_centered.shape[0], X_train_centered.shape[1])
    X_train_pca, X_test_pca, explained_sum = fast_pca_like_matlab(
        X_train_centered, X_test_centered, n_components
    )

    svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    svr.fit(X_train_pca, y_train)
    train_pred = svr.predict(X_train_pca)
    test_pred = svr.predict(X_test_pca)

    info = {
        "original_dim": original_dim,
        "selected_dim": selected_dim,
        "pca_dim": n_components,
        "explained": explained_sum,
    }
    return train_pred, test_pred, info


def run_multilayer_svr(base_dir: str | Path, cfg, debug: bool = True):
    base_dir = Path(base_dir)
    layer_names = load_layer_names(base_dir)
    y_train = load_scores(base_dir, "train")
    y_test = load_scores(base_dir, "test")

    all_layer_results = []
    for i, layer_name in enumerate(layer_names):
        X_train = load_feature(base_dir, "train", layer_name)
        X_test = load_feature(base_dir, "test", layer_name)

        train_pred, test_pred, info = run_single_layer_svr(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            k_value=cfg.k_value,
            n_components=cfg.n_components,
            preselect_ascending=cfg.preselect_ascending,
            C=cfg.layer_svr_c,
            gamma=cfg.layer_svr_gamma,
            epsilon=cfg.layer_svr_epsilon,
        )

        all_layer_results.append(
            {
                "idx": i,
                "layer_name": layer_name,
                "train_pred": train_pred,
                "test_pred": test_pred,
                "rmse": rmse(y_test, test_pred),
                "plcc": plcc(y_test, test_pred),
                "srcc": srcc(y_test, test_pred),
                "info": info,
            }
        )

    # 單模型 debug：不直接用 all layers 當唯一結果，而是先依單層表現排序做診斷
    by_plcc = sorted(all_layer_results, key=lambda x: x["plcc"], reverse=True)
    by_srcc = sorted(all_layer_results, key=lambda x: x["srcc"], reverse=True)

def run_multilayer_svr(base_dir: str | Path, cfg, debug: bool = True):
    base_dir = Path(base_dir)
    layer_names = load_layer_names(base_dir)
    y_train = load_scores(base_dir, "train")
    y_test = load_scores(base_dir, "test")

    # ------------------------------------------------------------
    # 1) split train -> inner_train / val
    # ------------------------------------------------------------
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_train))
    rng.shuffle(indices)

    val_ratio = 0.2
    n_val = max(1, int(round(len(indices) * val_ratio)))
    val_idx = indices[:n_val]
    inner_train_idx = indices[n_val:]

    y_inner_train = y_train[inner_train_idx]
    y_val = y_train[val_idx]

    layer_rank_results = []

    # ------------------------------------------------------------
    # 2) layer ranking using validation only
    # ------------------------------------------------------------
    for i, layer_name in enumerate(layer_names):
        X_train_full = load_feature(base_dir, "train", layer_name)

        X_inner_train = X_train_full[inner_train_idx]
        X_val = X_train_full[val_idx]

        inner_train_pred, val_pred, info = run_single_layer_svr(
            X_train=X_inner_train,
            X_test=X_val,
            y_train=y_inner_train,
            k_value=cfg.k_value,
            n_components=cfg.n_components,
            preselect_ascending=cfg.preselect_ascending,
            C=cfg.layer_svr_c,
            gamma=cfg.layer_svr_gamma,
            epsilon=cfg.layer_svr_epsilon,
        )

        layer_rank_results.append(
            {
                "idx": i,
                "layer_name": layer_name,
                "inner_train_pred": inner_train_pred,
                "val_pred": val_pred,
                "val_rmse": rmse(y_val, val_pred),
                "val_plcc": plcc(y_val, val_pred),
                "val_srcc": srcc(y_val, val_pred),
                "info": info,
            }
        )

    rank_by = getattr(cfg, "rank_by", "plcc")
    rank_key = "val_srcc" if str(rank_by).lower() == "srcc" else "val_plcc"

    layer_rank_results.sort(key=lambda x: x[rank_key], reverse=True)

    top_k = getattr(cfg, "top_k_layers", None)
    if top_k is None:
        selected_rank_results = layer_rank_results
    else:
        selected_rank_results = layer_rank_results[:top_k]

    selected_layer_names = [x["layer_name"] for x in selected_rank_results]

    # ------------------------------------------------------------
    # 3) re-fit selected layers on full train, then predict on train/test
    # ------------------------------------------------------------
    selected_final_layers = []
    for item in selected_rank_results:
        layer_name = item["layer_name"]

        X_train = load_feature(base_dir, "train", layer_name)
        X_test = load_feature(base_dir, "test", layer_name)

        train_pred, test_pred, info = run_single_layer_svr(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            k_value=cfg.k_value,
            n_components=cfg.n_components,
            preselect_ascending=cfg.preselect_ascending,
            C=cfg.layer_svr_c,
            gamma=cfg.layer_svr_gamma,
            epsilon=cfg.layer_svr_epsilon,
        )

        selected_final_layers.append(
            {
                "layer_name": layer_name,
                "train_pred": train_pred,
                "test_pred": test_pred,
                "test_rmse": rmse(y_test, test_pred),
                "test_plcc": plcc(y_test, test_pred),
                "test_srcc": srcc(y_test, test_pred),
                "rank_val_plcc": item["val_plcc"],
                "rank_val_srcc": item["val_srcc"],
                "info": info,
            }
        )

    # ------------------------------------------------------------
    # 4) high-level SVR on selected layers only
    # ------------------------------------------------------------
    meta_train = np.concatenate(
        [x["train_pred"].reshape(-1, 1) for x in selected_final_layers], axis=1
    )
    meta_test = np.concatenate(
        [x["test_pred"].reshape(-1, 1) for x in selected_final_layers], axis=1
    )

    high_svr = SVR(
        kernel="rbf",
        C=cfg.high_svr_c,
        gamma=cfg.high_svr_gamma,
        epsilon=cfg.high_svr_epsilon,
    )
    high_svr.fit(meta_train, y_train)
    final_pred = high_svr.predict(meta_test)

    result = {
        "layer_ranking_results": layer_rank_results,   # validation ranking info
        "layer_results": selected_final_layers,        # selected layers refit on full train
        "selected_layers": selected_layer_names,
        "final_pred": final_pred,
        "final_rmse": rmse(y_test, final_pred),
        "final_plcc": plcc(y_test, final_pred),
        "final_srcc": srcc(y_test, final_pred),
        "y_test": y_test,
    }

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: validation-based layer selection")
        print("=" * 80)

        print(f"\nRanking criterion: {rank_key}")
        print(f"Top-K selected: {len(selected_layer_names)}")

        print("\nTop 10 layers by validation ranking")
        for rank, item in enumerate(layer_rank_results[:10], start=1):
            info = item["info"]
            print(
                f"{rank:02d}. {item['layer_name']:<30} "
                f"VAL_PLCC={item['val_plcc']:.4f}  VAL_SRCC={item['val_srcc']:.4f}  VAL_RMSE={item['val_rmse']:.4f}  "
                f"[orig={info['original_dim']}, sel={info['selected_dim']}, pca={info['pca_dim']}]"
            )

        print("\nSelected layers")
        for rank, name in enumerate(selected_layer_names, start=1):
            print(f"{rank:02d}. {name}")

        print("\nSelected layers refit on full train -> test performance")
        for rank, item in enumerate(selected_final_layers, start=1):
            print(
                f"{rank:02d}. {item['layer_name']:<30} "
                f"TEST_PLCC={item['test_plcc']:.4f}  TEST_SRCC={item['test_srcc']:.4f}  TEST_RMSE={item['test_rmse']:.4f}"
            )

        print("\nFinal high-level fusion result")
        print(
            f"Fusion(top-{len(selected_layer_names)}) | "
            f"PLCC={result['final_plcc']:.4f}, "
            f"SRCC={result['final_srcc']:.4f}, "
            f"RMSE={result['final_rmse']:.4f}"
        )

        pred_std = np.std(final_pred)
        y_std = np.std(y_test)
        print(f"Prediction std = {pred_std:.4f}, Ground-truth std = {y_std:.4f}")

        print("=" * 80 + "\n")

    return result