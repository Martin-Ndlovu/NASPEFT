import sys
from time import time
import torch
from accelerate.utils import broadcast
import numpy as np
import random
import os
from adapterhub.random_search.search_space import AdapterSearchSpace
from adapterhub.random_search.base_function import Problem
from adapterhub.random_search.utils import get_pareto_points
import argparse
import logger as logging_setup
from accelerate import Accelerator
import logging
from definition import ROOT_DIR

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--method", type=str, default="random", choices=["random"])
parser.add_argument('-sp', "--save_path", type=str, default=f"{ROOT_DIR}/output/")
parser.add_argument('-dp', "--data_path", type=str, default=f"{ROOT_DIR}/datasets/wikitext/wikitext-2-v1/")
parser.add_argument('-mp', "--model_path", type=str, default=f"{ROOT_DIR}/models/Llama-3.2-1B/")
parser.add_argument('-t', "--task", type=str, default="wikitext")
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-mi', '--max_iter', type=int, default=200)
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('-ni', "--n_init", type=int, default=20)
parser.add_argument('-o', "--objectives", nargs="+", default=["param", "perplexity"])
parser.add_argument('--overwrite', action="store_true")
parser.add_argument('--mock_run', action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument('-an', "--adapter_name", type=str, default="naspeft")
parser.add_argument('-rd', "--resplit_dataset", type=bool, default=False)
args, _ = parser.parse_known_args()
logger = logging_setup.get_logger(__name__)
# logger.setLevel(logging.DEBUG)

accelerator = Accelerator()

# Handle relative paths
if args.save_path.startswith("./"):
    args.save_path = args.save_path.split("./")[1]
    args.save_path = os.path.join(ROOT_DIR, args.save_path)
if args.model_path.startswith("./"):
    args.model_path = args.model_path.split("./")[1]
    args.model_path = os.path.join(ROOT_DIR, args.model_path)

save_path = os.path.join(
    args.save_path, f"NAS_{args.task}_random_seed_{args.seed}_bs_{args.batch_size}"
)
data_path = args.data_path
model_path = args.model_path

resume = False
if not os.path.exists(save_path):
    os.makedirs(save_path)
elif args.resume:
    resume = True
    if accelerator.is_main_process:
        logger.info(f"Resuming from {save_path}")
elif not args.overwrite:
    raise FileExistsError(
        f"{save_path} is not empty. Change to another save_path, or enable the overwrite flag.")
logging_path = os.path.join(save_path, "train_logs.log")
logging_setup.setup_logging(logging_path, 'w')
if accelerator.is_main_process:
    logger.info(f"Save dir = {save_path}")
    logger.info(vars(args))
if args.mock_run and accelerator.is_main_process:
    logger.warning("This run is a mock run. No actual training will be performed.")

# Fix seeds
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def run_one_replication(
        dtype: torch.dtype = torch.float,
        device: str = None,
        save_frequency: int = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if accelerator.is_main_process:
        logger.info(f"[Rank {local_rank}] Running on device {device}" if local_rank != -1 else f"Single-GPU mode using {device}")

    iterations = max(1, args.max_iter // args.batch_size)
    batch_size = args.batch_size
    tkwargs = {"dtype": torch.bfloat16, "device": device}
    n_initial_points = args.n_init
    ss = AdapterSearchSpace(seed=seed)
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
        final_test=False,
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
            new_Z, new_X = list(zip(*[ss.sample_configuration(return_dict_repr=True)
                                    for _ in range(remaining_init)]))
            new_Z = torch.stack(
                [torch.from_numpy(z).to(**tkwargs) for z in new_Z])
            Z = torch.cat([Z, new_Z], dim=0)
            X += list(new_X)
            if accelerator.is_main_process:
                new_Y = f(new_X).to(**tkwargs)
            else:
                new_Y = torch.empty((remaining_init, f.num_objectives), **tkwargs)
            new_Y = broadcast(new_Y)
            Y = torch.cat([Y, new_Y])
        if accelerator.is_main_process:
            logger.info(f"Sampled Configurations (First 10 or available):\n{[str(X[i]) for i in range(min(10, len(X)))]}")
    else:
        Z, X = list(zip(*[ss.sample_configuration(return_dict_repr=True)
                        for _ in range(n_initial_points)]))
        X = list(X)
        Z = torch.stack([torch.from_numpy(z).to(**tkwargs) for z in Z])
        if accelerator.is_main_process:
            Y = f(X).to(**tkwargs)
        else:
            Y = torch.empty((n_initial_points, f.num_objectives), **tkwargs)
        Y = broadcast(Y)
        wall_time_prev = None
        if accelerator.is_main_process:
            logger.info(f"Sampled Configurations (First 10 or available):\n{[str(X[i]) for i in range(min(10, len(X)))]}")

    is_moo = f.num_objectives > 1
    if is_moo and f.ref_point is not None:
        ref_point = f.ref_point.to(**tkwargs)
        if accelerator.is_main_process:
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
        if accelerator.is_main_process:
            logger.info(
                f"Starting seed {seed}, iteration {i}, "
                f"time: {time() - start_time}, "
                f"Last obj: {Y[-batch_size:]}"
            )
        # Random Search
        z_temp, candidates_x = list(
            zip(*[ss.sample_configuration(return_dict_repr=True) for _ in range(batch_size)]))
        candidates_z = torch.stack(
            [torch.from_numpy(z).to(**tkwargs) for z in z_temp])
        try:
            if accelerator.is_main_process:
                new_y = f(candidates_x).to(**tkwargs)
            else:
                new_y = torch.empty((batch_size, f.num_objectives), **tkwargs)
            new_y = broadcast(new_y)
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Error evaluating batch: {e}")
            continue

        X += candidates_x
        Y = torch.cat([Y, new_y], dim=0)
        Z = torch.cat([Z, candidates_z], dim=0)
        wall_time[i] = time() - start_time

        # Save periodically
        if accelerator.is_main_process:
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
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        y = output_dict['Y']
        pareto_mask = get_pareto_points(y)
        non_dom_idx = torch.where(pareto_mask)[0]
        for inx in non_dom_idx:
            logger.info(f"Pareto point: {y[inx]}")
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