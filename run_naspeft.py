import sys
from time import time
import torch
import numpy as np
import random
import os
import json
from adapterhub.smbo.search_space import AdapterSearchSpace
from adapterhub.smbo.base_function import Problem
from adapterhub.smbo.utils import get_pareto_points
import argparse
import logger as logging_setup
import logging
from definition import ROOT_DIR
from adapterhub.surrogate.random_forest_surrogate import (
    RandomForestSurrogate,
    build_features,
    build_derived_features,
    load_layer_sensitivity,
)
# ── Layer-sensitivity probe ─────────────────────────────────────
from adapterhub.layer_selection import run_layer_selection
# ── Acquisition + exact parameter oracle ───────────────────────
from adapterhub.smbo.param_oracle import batch_param_percentage
from adapterhub.smbo.acquisition import (
    DNormStats,
    acquisition_scores,
    select_topB_lowest,
    draw_lambda,
)

# (NCCL / distributed env vars removed for single-GPU run)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--method", type=str, default="random", choices=["random"])
parser.add_argument('-sp', "--save_path", type=str, default=f"{ROOT_DIR}/output/random_layer_selection/")
parser.add_argument('-dp', "--data_path", type=str, default=f"{ROOT_DIR}/datasets/glue/")
parser.add_argument('-mp', "--model_path", type=str, default=f"{ROOT_DIR}/models/bert-base-uncased/")
parser.add_argument('-t', "--task", type=str, default="mrpc")
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-mi', '--max_iter', type=int, default=180)
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('-ni', "--n_init", type=int, default=20)
parser.add_argument('-o', "--objectives", nargs="+", default=["param", "acc"])
parser.add_argument('--overwrite', action="store_true")
parser.add_argument('--mock_run', action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument('-an', "--adapter_name", type=str, default="naspeft")
parser.add_argument('-rd', "--resplit_dataset", type=bool, default=False)
parser.add_argument('-cr', "--custom_split", type=bool, default=True)
parser.add_argument('--random_search', action="store_true",
                    help="Ablation: bypass surrogate + acquisition, use random search.")
# Exploration constant kappa for the LCB term in the acquisition.
parser.add_argument('-k', "--kappa", type=float, default=1.0,
                    help="Exploration constant for the LCB term in eq. 7.")
# ── Layer-selection controls ──────────────────────────────────────────
parser.add_argument("--skip_layer_selection", action="store_true",
                    help="Skip the probe and search over ALL physical "
                         "layers (no reduction).")
parser.add_argument("--random_layer_selection", action="store_true",
                    help="Skip the probe and randomly select a subset of "
                         "physical layers to search over (no reduction).")
parser.add_argument("--num_random_layers", type=int, default=None)
parser.add_argument("--layer_selection_summary", type=str, default=None,
                    help="Path to an existing layer_selection_summary.json to "
                         "reuse instead of re-running the probe.")
parser.add_argument("--force_layer_selection", action="store_true",
                    help="Re-run the probe even if cached results exist for "
                         "this (model, dataset).")
parser.add_argument("--ls_epochs", type=int, default=10,
                    help="Epochs per run for the layer-selection probe.")
# ── Parameter budget control ─────────────────────────────────────────────────
parser.add_argument('-pb', "--param_budget", type=float, default=None,
                    help="Optional upper bound on parameter percentage (param%%). "
                         "If set, only configurations with param%% strictly below "
                         "this budget (per the exact param oracle) are considered "
                         "for the initial design and the per-iteration candidate "
                         "pool, for both --n_init and during search. Ignored (with "
                         "a warning) if the param oracle is unavailable for this "
                         "backbone.")
args, _ = parser.parse_known_args()
logger = logging_setup.get_logger(__name__)

# Handle relative paths and add model name to save_path for organization. The probe uses the same logic to determine where to write its output, so this also ensures the probe and main run stay in sync on the filesystem.
if args.save_path.startswith("./"):
    args.save_path = args.save_path.split("./")[1]
    args.save_path = os.path.join(ROOT_DIR, args.save_path)
if args.model_path.startswith("./"):
    args.model_path = args.model_path.split("./")[1]
    args.model_path = os.path.join(ROOT_DIR, args.model_path)

model_name = os.path.basename(args.model_path)
save_path = os.path.join(
    args.save_path, f"{model_name}_{args.task}_random_seed_{args.seed}_bs_{args.batch_size}"
)
data_path = os.path.join(
    args.data_path, args.task
) if "glue" in args.data_path else args.data_path
model_path = args.model_path

resume = False
if not os.path.exists(save_path):
    os.makedirs(save_path)
elif args.resume:
    resume = True
    logger.info(f"Resuming from {save_path}")
elif not args.overwrite:
    raise FileExistsError(
        f"{save_path} is not empty. Change to another save_path, or enable the overwrite flag.")
logging_path = os.path.join(save_path, "train_logs.log")
logging_setup.setup_logging(logging_path, 'w')
logger.info(f"Save dir = {save_path}")
logger.info(vars(args))
if args.mock_run:
    logger.warning("This run is a mock run. No actual training will be performed.")

# Fix seeds
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Dedicated RNG for the per-iteration lambda draws (reproducible, independent
# of the global numpy stream so adding the acquisition does not perturb the
# configuration-sampling sequence relative to older runs with the same seed).
_acq_rng = np.random.default_rng(seed)


def sample_configurations_with_budget(
        ss,
        n,
        param_budget=None,
        oracle_available=False,
        model_path=None,
        num_layers=None,
        logger=None,
        max_resample_factor=20,
        oversample_factor=2,
):
    """Sample `n` configurations from the search space, optionally enforcing
    a parameter-budget constraint via the exact param oracle.

    If `param_budget` is None or the param oracle is unavailable for this
    backbone, this is equivalent to the original unfiltered sampling:
    `n` independent calls to `ss.sample_configuration(return_dict_repr=True)`.

    Otherwise, configurations are drawn in (oversampled) batches and scored
    with `batch_param_percentage`; only configurations with
    `param% < param_budget` are accepted. This repeats until `n` accepted
    configurations have been collected, or `max_resample_factor * n` total
    configurations have been drawn (whichever comes first).

    If the budget is too tight to find `n` in-budget configurations within
    that draw cap, the shortfall is filled with the lowest-param%
    configurations seen among the rejected draws (so the caller always gets
    exactly `n` configurations back), and a warning is logged.

    Returns
    -------
    (Z, X) : tuple of lists
        Z: list of np.ndarray encodings, as returned by
           `ss.sample_configuration(return_dict_repr=True)`.
        X: list of dict configuration representations.
    """
    if param_budget is None or not oracle_available:
        z_list, x_list = zip(*[
            ss.sample_configuration(return_dict_repr=True) for _ in range(n)
        ])
        return list(z_list), list(x_list)

    accepted_z, accepted_x = [], []
    leftover_z, leftover_x, leftover_c = [], [], []
    total_drawn = 0
    max_total = max(n, max_resample_factor * n)

    while len(accepted_z) < n and total_drawn < max_total:
        remaining = n - len(accepted_z)
        batch_n = min(max(remaining * oversample_factor, remaining), max_total - total_drawn)
        z_batch, x_batch = zip(*[
            ss.sample_configuration(return_dict_repr=True) for _ in range(batch_n)
        ])
        total_drawn += batch_n

        c_batch = batch_param_percentage(
            list(x_batch), model_path=model_path, num_layers=num_layers
        )

        for z, x, c in zip(z_batch, x_batch, c_batch):
            if c < param_budget:
                accepted_z.append(z)
                accepted_x.append(x)
                if len(accepted_z) == n:
                    break
            else:
                leftover_z.append(z)
                leftover_x.append(x)
                leftover_c.append(c)

    if len(accepted_z) < n:
        shortfall = n - len(accepted_z)
        if logger is not None:
            logger.warning(
                f"param_budget={param_budget}: only found {len(accepted_z)}/{n} "
                f"configurations with param% < {param_budget} after drawing "
                f"{total_drawn} samples. Filling the remaining {shortfall} "
                f"slot(s) with the lowest-param% configurations seen "
                f"(these may exceed the budget)."
            )
        order = np.argsort(leftover_c)[:shortfall]
        for idx in order:
            accepted_z.append(leftover_z[idx])
            accepted_x.append(leftover_x[idx])

        # Last-resort fallback: if even the leftovers can't cover the
        # shortfall (e.g. every drawn sample was already accepted), top up
        # with plain unfiltered samples so the caller always gets `n` back.
        while len(accepted_z) < n:
            z, x = ss.sample_configuration(return_dict_repr=True)
            accepted_z.append(z)
            accepted_x.append(x)

    return accepted_z, accepted_x


def run_one_replication(
        dtype: torch.dtype = torch.float,
        device: str = None,
        save_frequency: int = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"GPU mode using {device}")

    iterations = max(1, args.max_iter // args.batch_size)
    batch_size = args.batch_size
    # Use fp32 for tensor storage to avoid bf16 saturation issues
    tkwargs = {"dtype": torch.float32, "device": device}

    # Set up search space, surrogate, and problem
    n_initial_points = args.n_init
    is_large = "-large" in model_path.lower()
    is_t5 = "t5" in model_path.lower()
    is_llama = "llama" in model_path.lower()

    # ── Stage: layer-sensitivity probe → G1 (searchable layers) ─────────
    # Runs (or reuses) the probe, then passes G1 + the physical layer count
    # EXPLICITLY into AdapterSearchSpace. The search space carries only |G1|
    # mask variables internally but EMITS full physical-width configs, so no
    # downstream component (Problem / nas_search_plus / param_oracle /
    # surrogate) needs any change.
    if args.skip_layer_selection:
        g1 = None  # search over all physical layers
        num_physical_layers = None  # AdapterSearchSpace infers from is_large/is_llama
        logger.info("Layer selection SKIPPED — searching over all physical layers.")
    elif args.random_layer_selection:
        if is_llama:
            num_physical_layers = 16
        elif is_large:
            num_physical_layers = 24
        else:
            num_physical_layers = 12
        k = args.num_random_layers or num_physical_layers
        if not (1 <= k <= num_physical_layers):
            raise ValueError(f"--num_random_layers={k} out of range for {num_physical_layers}")
        _layer_rng = random.Random(args.seed)
        g1 = sorted(_layer_rng.sample(range(num_physical_layers), k))
        logger.info(f"RANDOM layer selection: |G1|={k} of {num_physical_layers} | G1={g1}")
    else:
        # Deterministic path the probe writes to (mirrors run_layer_selection's
        # output_dir = output_root/<basename(model)>_<dataset>).
        _ls_root = os.path.join(save_path, "layer_selection")
        _auto_summary = os.path.join(
            _ls_root,
            f"{os.path.basename(model_path)}_{args.task}",
            "layer_selection_summary.json",
        )
        if args.layer_selection_summary and os.path.exists(args.layer_selection_summary):
            with open(args.layer_selection_summary, "r") as fp:
                ls_result = json.load(fp)
            logger.info(
                f"Loaded layer selection from {args.layer_selection_summary}"
            )
        elif os.path.exists(_auto_summary) and not args.force_layer_selection:
            # Probe already ran for this (model, dataset) — reuse it, skip the
            # expensive (num_layers + 4)-run probe entirely.
            with open(_auto_summary, "r") as fp:
                ls_result = json.load(fp)
            logger.info(
                f"Reusing existing layer-selection results: {_auto_summary} "
                f"(probe skipped; delete this file or pass "
                f"--force_layer_selection to re-run)"
            )
        else:
            logger.info("Running layer-sensitivity probe...")
            ls_result = run_layer_selection(
                model_name_or_path=model_path,
                dataset_name=args.task,
                output_root=_ls_root,
                epochs=args.ls_epochs,
                seed=args.seed,
                glue_local_root=(
                    os.path.dirname(data_path) if "glue" in args.data_path else None
                ),
            )
        g1 = ls_result["searchable_layers"]
        num_physical_layers = ls_result["num_layers"]
        logger.info(
            f"Layer selection: tau_{ls_result.get('selected_percentile')} | "
            f"|G1|={ls_result['num_searchable']} of {num_physical_layers} | "
            f"G1={g1} | G2(frozen)={ls_result['frozen_layers']}"
        )

        # ── Sync the surrogate's sensitivity file from the SAME probe ────────
        # The surrogate's importance feature uses per_layer_raw_metric (the
        # full physical-depth per-layer probe performance: accuracy for GLUE,
        # perplexity for causal LM). Write it into layer_sensitivity.json
        # keyed [model_name][task_name] using EXACTLY the normalization
        # load_layer_sensitivity applies, so the surrogate always consumes the
        # same probe data that produced G1 (no manual sync, no drift).
        try:
            _raw_vec = ls_result["per_layer_raw_metric"]
            if len(_raw_vec) != num_physical_layers:
                logger.warning(
                    f"per_layer_raw_metric length {len(_raw_vec)} != "
                    f"num_layers {num_physical_layers}; writing anyway."
                )
            _mn = args.model_path
            if "/" in _mn:
                _mn = _mn.split("/")[-2] if _mn.endswith("/") else _mn.split("/")[-1]
            _sens_path = os.path.join(
                ROOT_DIR, "adapterhub/surrogate/layer_sensitivity.json"
            )
            os.makedirs(os.path.dirname(_sens_path), exist_ok=True)
            if os.path.exists(_sens_path):
                with open(_sens_path, "r") as fp:
                    _sens_data = json.load(fp)
            else:
                _sens_data = {}
            _sens_data.setdefault(_mn, {})
            _sens_data[_mn][args.task] = [float(v) for v in _raw_vec]
            with open(_sens_path, "w") as fp:
                json.dump(_sens_data, fp, indent=2)
            logger.info(
                f"Wrote layer_sensitivity.json[{_mn}][{args.task}] "
                f"({len(_raw_vec)} values) from probe per_layer_raw_metric."
            )
        except Exception as e:
            logger.warning(
                f"Could not auto-write layer_sensitivity.json: {e}. "
                f"Surrogate will fall back to whatever is already on disk."
            )

    ss = AdapterSearchSpace(
        seed=seed, is_large=is_large, is_t5=is_t5, is_llama=is_llama,
        g1=g1, num_physical_layers=num_physical_layers,
    )
    # The width of every EMITTED config (physical layers) — drives the
    # surrogate's active-layer derived feature. This is NOT |G1|; it is the
    # full transformer depth, because encode_config emits the physical mask.
    physical_num_layers = ss.num_physical_layers
    logger.info(
        f"Search space: {ss.num_search_layers} searchable layer vars "
        f"-> emits {physical_num_layers}-dim physical configs."
    )

    surrogate = RandomForestSurrogate()
    # The surrogate's importance feature needs a full physical-depth sensitivity
    # vector. When layer selection ran (any branch) it was auto-written above.
    # On --skip_layer_selection nothing wrote it, so fall back gracefully to a
    # neutral (all-ones) vector of the correct physical width: the importance
    # feature then degenerates to "active-layer count" (a valid, if weaker,
    # structural feature) instead of crashing or using a stale/mismatched file.
    try:
        layer_sensitivity = load_layer_sensitivity(args.model_path, args.task)
        if len(layer_sensitivity) != physical_num_layers:
            logger.warning(
                f"layer_sensitivity length {len(layer_sensitivity)} != "
                f"physical layers {physical_num_layers}; using neutral vector."
            )
            layer_sensitivity = np.ones(physical_num_layers, dtype=float)
    except Exception as e:
        logger.warning(
            f"load_layer_sensitivity failed ({e}); using neutral all-ones "
            f"vector of width {physical_num_layers} (importance feature "
            f"degenerates to active-layer count)."
        )
        layer_sensitivity = np.ones(physical_num_layers, dtype=float)
    f = Problem(
        adapter_name=args.adapter_name,
        task_name=args.task,
        search_space=ss,
        save_path=save_path,
        data_path=data_path,
        model_path=model_path,
        objectives=args.objectives,
        logger=logger,
        seed=args.seed,
        mock_run=args.mock_run,
        resplit_dataset=args.resplit_dataset,
        custom_split=args.custom_split,
        is_large=is_large,
        final_test=False,
    )

    # ── Performance-axis sign for the acquisition's minimization convention ──
    # GLUE (accuracy / MCC / Spearman): higher is better -> negate normalized
    #   performance inside eq. 7 so "lower alpha = better".
    # LLaMA / wikitext (perplexity): lower is better already -> no negation.
    glue_higher_is_better = not (is_llama or "wikitext" in args.task.lower())

    # ── Search-space maximum param% (normalizes c_tilde to [0,1] in eq. 7) ──
    # Largest configuration: every layer active, largest LoRA rank, the
    # smallest parallel reduction factor (=1 -> largest bottleneck b=H), and
    # the LONGEST prefix (reduction_prefix is the prefix LENGTH Lp, so larger
    # = more params; the search-space max is 2**10 = 1024, or 2**11 for llama).
    # Computed exactly via the oracle so c_tilde is bounded in [0,1].
    num_layers_full = ss._get_num_layers()
    # In run_naspeft.py, replace the _max_log2 line:
    if is_llama:
        _max_log2 = 11
    elif is_t5 and not is_large:
        _max_log2 = 9    # t5-base search space cap
    else:
        _max_log2 = 10   # bert, roberta, t5-large
    max_cfg = {f"leave_out_{i}": 0 for i in range(num_layers_full)}
    max_cfg["lora_r"] = 64
    max_cfg["lora_alpha"] = 32
    max_cfg["reduction_parallel"] = 1            # smallest factor -> largest bottleneck
    max_cfg["reduction_prefix"] = 2 ** _max_log2  # longest prefix -> most params
    try:
        c_max = batch_param_percentage(
            [max_cfg], model_path=model_path, num_layers=num_layers_full
        )[0]
        oracle_available = True
        logger.info(f"Acquisition eq.7: kappa={args.kappa}, c_max={c_max:.6f}, "
                    f"glue_higher_is_better={glue_higher_is_better}")
    except ValueError as _oracle_err:
        c_max = 1.0
        oracle_available = False
        logger.warning(
            f"Param oracle not available for this backbone ({_oracle_err}). "
            f"Acquisition eq.7 DISABLED — falling back to surrogate-greedy selection."
        )

    # ── Parameter budget control ─────────────────────────────────────────────
    if args.param_budget is not None:
        if not oracle_available:
            logger.warning(
                f"--param_budget={args.param_budget} requested but the param "
                f"oracle is unavailable for this backbone; budget filtering "
                f"will be DISABLED (no effect)."
            )
        else:
            logger.info(
                f"Parameter budget control ENABLED: only configurations with "
                f"param% < {args.param_budget} will be considered for the "
                f"initial design and the per-iteration candidate pool."
            )

    # Generate initial data or load from previous run
    if resume:
        with open(os.path.join(save_path, "result_stats.pt"), "rb") as fp:
            load_dict = torch.load(fp, map_location="cpu")
        Z = load_dict["Z"].to(**tkwargs)
        X = load_dict["X"]
        Y = load_dict["Y"].to(**tkwargs)
        wall_time_prev = load_dict["wall_time"]
        if len(X) < n_initial_points:
            remaining_init = n_initial_points - len(X)
            new_Z, new_X = sample_configurations_with_budget(
                ss, remaining_init,
                param_budget=args.param_budget,
                oracle_available=oracle_available,
                model_path=model_path,
                num_layers=num_layers_full,
                logger=logger,
            )
            new_Z = torch.stack(
                [torch.from_numpy(z).to(**tkwargs) for z in new_Z])
            Z = torch.cat([Z, new_Z], dim=0)
            X += list(new_X)
            new_Y = f(new_X).to(**tkwargs)
            Y = torch.cat([Y, new_Y])
        logger.info(
            f"Sampled Configurations (First 10 or available):\n"
            f"{[str(X[i]) for i in range(min(10, len(X)))]}"
        )
    else:
        Z, X = sample_configurations_with_budget(
            ss, n_initial_points,
            param_budget=args.param_budget,
            oracle_available=oracle_available,
            model_path=model_path,
            num_layers=num_layers_full,
            logger=logger,
        )
        X = list(X)
        Z = torch.stack([torch.from_numpy(z).to(**tkwargs) for z in Z])
        Y = f(X).to(**tkwargs)
        wall_time_prev = None
        logger.info(
            f"Sampled Configurations (First 10 or available):\n"
            f"{[str(X[i]) for i in range(min(10, len(X)))]}"
        )

    # Prepare data for surrogate training.
    # Z is now PHYSICAL-width (encode_config emits the full physical leave_out
    # mask + module HPs), so the derived-feature slicing must use the physical
    # layer count, NOT len(layer_sensitivity) (which is keyed to |G1|).
    Z_np = Z.detach().cpu().float().numpy()
    num_layers = physical_num_layers
    derived_features = build_derived_features(Z_np, layer_sensitivity, num_layers=num_layers)
    X_surrogate = build_features(Z_np, derived_features)
    y_surrogate = Y.detach().cpu().float().numpy()
    y_surrogate = y_surrogate[:, 1].reshape(-1, 1)  # Use only accuracy for surrogate targets

    logger.info(f"Training surrogate on {len(X_surrogate)} initial points...")
    logger.info(f"Initial surrogate features shape: {X_surrogate.shape}")
    logger.info(f"Initial surrogate targets shape: {y_surrogate.shape}")
    logger.info(f"Sample surrogate features (First 5 or available):\n{X_surrogate[:5]}")
    logger.info(f"Sample surrogate targets (First 5 or available):\n{y_surrogate[:5]}")

    is_moo = f.num_objectives > 1
    if is_moo and f.ref_point is not None:
        ref_point = f.ref_point.to(**tkwargs)
        logger.info(f"Reference point = {ref_point}")
    else:
        ref_point = None

    # Set counters
    start_time = time()
    existing_iterations = len(X) // batch_size
    wall_time = torch.zeros(iterations, dtype=dtype)
    if wall_time_prev is not None:
        wall_time[:existing_iterations] = wall_time_prev.view(-1)

    for i in range(existing_iterations, iterations):
        logger.info(
            f"Starting seed {seed}, iteration {i}, "
            f"time: {time() - start_time}, "
            f"Last obj: {Y[-batch_size:]}"
        )

        if args.random_search:
            # ---- Random-search ablation: bypass surrogate + acquisition ----
            z_batch, x_batch = sample_configurations_with_budget(
                ss, batch_size,
                param_budget=args.param_budget,
                oracle_available=oracle_available,
                model_path=model_path,
                num_layers=num_layers,
                logger=logger,
            )
            candidates_z = torch.stack([torch.from_numpy(z).to(**tkwargs) for z in z_batch])
            candidates_x = list(x_batch)
            logger.info(f"Random search: drew {batch_size} candidates uniformly (no surrogate).")
        else:
            # Train surrogate
            surrogate.fit(X_surrogate, y_surrogate)

            # ---- Surrogate-guided candidate selection (paper eq. 7) ----
            candidate_pool = 200
            z_pool, x_pool = sample_configurations_with_budget(
                ss, candidate_pool,
                param_budget=args.param_budget,
                oracle_available=oracle_available,
                model_path=model_path,
                num_layers=num_layers,
                logger=logger,
            )
            z_pool_tensor = torch.stack(
                [torch.from_numpy(z).to(**tkwargs) for z in z_pool]
            )

            z_pool_np = z_pool_tensor.detach().cpu().float().numpy()
            derived_pool = build_derived_features(z_pool_np, layer_sensitivity, num_layers=num_layers)
            X_pool = build_features(z_pool_np, derived_pool)

            preds, uncert = surrogate.predict(X_pool)
            preds = preds.reshape(-1)
            uncert = np.asarray(uncert, dtype=float).reshape(-1)

            if oracle_available:
                c_pool = np.asarray(
                    batch_param_percentage(list(x_pool), model_path=model_path, num_layers=num_layers),
                    dtype=float,
                )
                d_stats = DNormStats.from_targets(y_surrogate)
                lambda_t = draw_lambda(_acq_rng)
                alpha = acquisition_scores(
                    pred_perf=preds, pred_std=uncert, param_pct=c_pool,
                    lambda_t=lambda_t, kappa=args.kappa, d_stats=d_stats,
                    c_max=c_max, glue_higher_is_better=glue_higher_is_better,
                )
                best_idx = select_topB_lowest(alpha, batch_size)
                logger.info(f"Acquisition eq.7: lambda={lambda_t:.4f}, selected top-{batch_size} by lowest alpha")
            else:
                if glue_higher_is_better:
                    best_idx = np.argsort(preds)[-batch_size:]
                else:
                    best_idx = np.argsort(preds)[:batch_size]
                logger.info(f"Surrogate-greedy (oracle disabled): pred_perf={preds[best_idx]}")

            candidates_x = [x_pool[j] for j in best_idx]
            candidates_z = z_pool_tensor[best_idx]

        try:
            new_y = f(candidates_x).to(**tkwargs)
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            continue

        X += candidates_x
        Y = torch.cat([Y, new_y], dim=0)
        Z = torch.cat([Z, candidates_z], dim=0)

        # ---- Update surrogate dataset ----
        new_Z_np = candidates_z.detach().cpu().float().numpy()
        new_derived = build_derived_features(new_Z_np, layer_sensitivity, num_layers=num_layers)
        new_X_surrogate = build_features(new_Z_np, new_derived)

        new_y_surrogate = new_y.detach().cpu().float().numpy()
        new_y_surrogate = new_y_surrogate[:, 1].reshape(-1, 1)  # accuracy only

        # Append
        X_surrogate = np.vstack([X_surrogate, new_X_surrogate])
        y_surrogate = np.vstack([y_surrogate, new_y_surrogate])

        wall_time[i] = time() - start_time

        # Save periodically
        if save_frequency is not None and (i + 1) % save_frequency == 0:
            output_dict = {
                "Z": Z.detach().cpu(),
                "X": X,
                "Y": Y.detach().cpu(),
                "wall_time": wall_time[:i + 1],
                "pareto_mask": get_pareto_points(Y).cpu(),
            }
            with open(os.path.join(save_path, "result_stats.pt"), "wb") as fp:
                torch.save(output_dict, fp)

    # Save final output
    output_dict = {
        "Z": Z.detach().cpu(),
        "X": X,
        "Y": Y.detach().cpu(),
        "wall_time": wall_time,
        "pareto_mask": get_pareto_points(Y).cpu(),
    }
    with open(os.path.join(save_path, "result_stats.pt"), "wb") as fp:
        torch.save(output_dict, fp)

    # Test Pareto-optimal points
    y = output_dict['Y']                       # stored Y (maximization convention)
    is_llama = "llama" in model_path.lower()

    # Build a tensor where BOTH objectives are "lower is better" for get_pareto_points.
    Y_for_pareto = y.clone()

    # --- column 0 : parameter % --------------------------------------------------
    # GLUE stored it as -param% (because maximization=1 -> (-1)**1). Flip back to
    # +param% so that MINIMIZING selects the parameter-efficient configs.
    # LLaMA stored it as +param% already (maximization=0) -> leave as is.
    if not is_llama:
        Y_for_pareto[:, 0] = -y[:, 0]          # -(-param%) = +param%
    # else: already +param%, no change.

    # --- column 1 : task performance --------------------------------------------
    if "wikitext" in args.task.lower() or is_llama:
        # Perplexity: lower is better already. Stored as +perplexity -> keep.
        pass
    else:
        # GLUE accuracy / MCC / Spearman: higher is better. Stored as +metric.
        # Negate so that MINIMIZING maximizes the metric.
        Y_for_pareto[:, 1] = -y[:, 1]

    pareto_mask = get_pareto_points(Y_for_pareto)
    non_dom_idx = torch.where(pareto_mask)[0]

    logger.info(
        f"Pareto front for final test: {len(non_dom_idx)} configs "
        f"(param-efficient AND high-performance trade-offs retained)"
    )

    for inx in non_dom_idx:
        logger.info(f"Pareto point (stored Y): {y[inx]}")
        logger.info(f"Point architecture: {X[inx]}")
        results_list = []
        for test_seed in range(40, 45):
            save_test_path = os.path.join(save_path, f"test/seed_{test_seed}/")
            f_result = Problem(
                adapter_name=args.adapter_name,
                task_name=args.task,
                search_space=ss,
                save_path=save_test_path,
                data_path=data_path,
                model_path=model_path,
                objectives=args.objectives,
                logger=logger,
                seed=test_seed,
                mock_run=args.mock_run,
                resplit_dataset=args.resplit_dataset,
                custom_split=args.custom_split,
                final_test=True,
            )
            try:
                Y_result = f_result(X[inx]).to(**tkwargs)
                results_list.append(Y_result)
            except Exception as e:
                logger.error(f"Error testing config {inx}: {e}")
                continue
        if results_list:
            logger.info(f"Test results: {results_list}")
            logger.info(f"Test results mean: {torch.mean(torch.stack(results_list), dim=0)}")
            logger.info(f"Test results std: {torch.std(torch.stack(results_list), dim=0)}")


if __name__ == "__main__":
    run_one_replication()