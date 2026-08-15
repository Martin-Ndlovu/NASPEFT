#------------------------------------------------------------------------------------------------
# THIS FILE IS PART OF THE NASPEFT FRAMEWORK AND IT DEFINES THE BASE FUNCTION FOR SEARCH.
# IT INCLUDES THE PROBLEM CLASS WHICH HANDLES EVALUATION OF CONFIGURATIONS.
#------------------------------------------------------------------------------------------------

import json
import random
import shutil
import sys
import logger
import subprocess
from datetime import timedelta
import os
from .search_space import AdapterSearchSpace
import ConfigSpace as CS
from typing import Optional, Union, Dict, Any, List
from settings import TASK_SETTINGS, TASK_SETTINGS_SO
import torch
from definition import ROOT_DIR


#----------------------------------------------------------------------------
# Problem Class Definition
#----------------------------------------------------------------------------
class Problem:
    def __init__(
            self,
            adapter_name: str,
            task_name: str,
            search_space: AdapterSearchSpace,
            save_path: str,
            data_path: str,
            model_path: str,
            noise_std: float = 1e-4,
            objectives: List[str] = None,
            is_large: bool = False,
            maximization: bool = True,
            logger=None,
            seed: int = None,
            resplit_dataset: bool = False,
            custom_split: bool = False,
            final_test: bool = False,
            mock_run: bool = False) -> None:
        self.adapter_name = adapter_name
        self.task_name = task_name
        self.is_large = is_large
        self.final_test = final_test
        self.objectives = objectives or TASK_SETTINGS[self.task_name].get("objectives", None)
        if len(self.objectives) == 1:
            TASK_SETTINGS[self.task_name]['ref_point'] = TASK_SETTINGS_SO[self.task_name]['ref_point']
            TASK_SETTINGS[self.task_name]['objectives'] = TASK_SETTINGS_SO[self.task_name]['objectives']
            logger.info(f"Task settings: {TASK_SETTINGS[self.task_name]}")
        logger.info(f"Optimizing for objectives: {self.objectives}")
        if self.objectives is None:
            raise ValueError("Objectives must be specified in settings or passed directly.")
        self.num_objectives = len(self.objectives)
        self.save_path = save_path
        self.data_path = data_path
        self.model_path = model_path
        self.logger = logger.info if logger else print
        self.search_space = search_space
        self.seed = seed
        self.maximization = int(maximization)
        if "llama" in model_path.lower():
            self.maximization = 0  # For LLaMA, we want to minimize both parameters and perplexity
        self.mock_run = mock_run
        self.resplit_dataset = resplit_dataset
        self.custom_split = custom_split
        if self.num_objectives > 1:
            ref_point = TASK_SETTINGS[self.task_name].get("ref_point", None)
            assert len(ref_point) == self.num_objectives
            for i, element in enumerate(ref_point):
                if self.objectives[i] == "param":
                    ref_point[i] *= float(((-1) ** self.maximization))
                else:  # perplexity
                    ref_point[i] *= float((-1) ** (self.maximization + 1))
            self.ref_point = torch.tensor(ref_point)
        else:
            self.ref_point = None
        self.noise_std = noise_std

    #--------------------------------------------------------------------
    # Single Configuration Evaluation
    #--------------------------------------------------------------------
    def _evaluate_single(self, config: Union[CS.Configuration, Dict[str, Any]], optim_kwargs: Optional[Dict[str, Any]] = None):
        assert self.task_name in TASK_SETTINGS, f"{self.task_name} not in TASK_SETTINGS."
        options = TASK_SETTINGS[self.task_name].copy()
        options.update(optim_kwargs or {})
        patience = options["patience"]
        if self.final_test:
            patience = 10
        num_epochs = options["num_epochs"]
        if self.final_test:
            num_epochs = 20
        save_type = options["save_type"]
        num_steps_per_save = options["num_steps_per_save"]
        logging_steps = options["logging_steps"]
        low_resource = options.get("low_resource", None)
        if isinstance(config, CS.Configuration):
            config_dict = self.search_space.config_to_dict(config)
        else:
            config_dict = config
        config_id = self.search_space.get_config_id(config)
        save_path_this_config = os.path.join(self.save_path, config_id)
        if os.path.exists(save_path_this_config):
            shutil.rmtree(save_path_this_config)
        os.makedirs(save_path_this_config, exist_ok=True)

        save_file = os.path.join(save_path_this_config, "config.json")

        with open(save_file, "w", encoding="utf8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        pass_output_dir = save_path_this_config.rstrip('/')
        self.logger(f"Start training config ID = {config_id}")

        #--------------------------------------------------------------------
        # Prepare and Execute Training Commands 
        #--------------------------------------------------------------------
      
        predict_flag = " --do_predict" if self.final_test else ""
        file_to_run = "nas_search_T5.py" if "t5" in self.model_path.lower() else "nas_search_plus.py"
        self.logger(f"Using file {file_to_run} for training.")
        commands = f"""
        cd {ROOT_DIR}/adapterhub
        python {file_to_run}\
        --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
        --do_train --do_eval --resplit_dataset --max_seq_length=128 --per_device_train_batch_size=32\
        --learning_rate=1e-4 --max_grad_norm=1.0 --warmup_ratio=0.0 --weight_decay=0.0\
        --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
        --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={logging_steps}\
        --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name} --custom_split={self.custom_split}{predict_flag}
        """ 
        commands_resplit = f"""
        cd {ROOT_DIR}/adapterhub
        python {file_to_run}\

        --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
        --do_train --do_eval --resplit_dataset --max_seq_length=128 --per_device_train_batch_size=32\
        --learning_rate=1e-4 --max_grad_norm=1.0 --warmup_ratio=0.0 --weight_decay=0.0\
        --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
        --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={logging_steps}\
        --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name} --custom_split={self.custom_split} {predict_flag}
        """
        commands_low_resource = f"""
        cd {ROOT_DIR}/adapterhub
        python {file_to_run}\
        --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
        --do_train --do_eval --max_seq_length=128 --per_device_train_batch_size=32\
        --learning_rate=1e-4 --max_grad_norm=1.0 --warmup_ratio=0.0 --weight_decay=0.0\
        --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
        --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={logging_steps}\
        --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name} --custom_split={self.custom_split} --max_train_samples={low_resource}{predict_flag}
        """

        #--------------------------------------------------------------------
        # Execute the appropriate command based on resource settings
        #--------------------------------------------------------------------
        if low_resource is not None:
            subprocess.call(commands_low_resource, shell=True)
        elif not self.resplit_dataset:
            subprocess.call(commands, shell=True)
        else:
            subprocess.call(commands_resplit, shell=True)

        #--------------------------------------------------------------------
        # Load Results and Compute Objectives
        #--------------------------------------------------------------------
        trainer_state = json.load(
            open(os.path.join(save_path_this_config, "trainer_state.json")))
        metrics = trainer_state["log_history"]
        params = json.load(
            open(os.path.join(save_path_this_config, "model_param_dict.json")))
        params_adapter = params["adapters"]

        # Ignore the above calculations and use the true number of parameters instead
        param = float(params_adapter) / float(params['model']) * 100. * float(((-1) ** self.maximization))

        # On final-test passes, prefer metrics computed on the held-out test
        # split (test_results.json, written by --do_predict
        # block when it has labels to score against, e.g. --resplit_dataset).
        # If that file is missing or doesn't contain a usable metric for this
        # task, fall back to the best validation metric from trainer_state.json
        # (the same source used during search).
        test_metrics = None
        if self.final_test:
            test_results_path = os.path.join(save_path_this_config, "test_results.json")
            if os.path.exists(test_results_path):
                try:
                    with open(test_results_path) as fp:
                        test_metrics = json.load(fp)
                except Exception as e:
                    self.logger(f"Could not read test_results.json ({e}); "
                                f"falling back to trainer_state.json eval metrics.")
                    test_metrics = None

        best_f1 = 0.0
        best_perplexity = float('inf')
        find_best = True

        # Calculate based on model used, llama should use perplexity, while others should use accuracy
        if "llama" in self.model_path.lower():
            for metric in metrics:
                if "eval_perplexity" in metric:
                    if metric['eval_perplexity'] < best_perplexity:
                        best_perplexity = metric['eval_perplexity']
            self.logger(f"Best perplexity: {best_perplexity}")
            best_perplexity *= float((1) ** (self.maximization + 1))
            all_objectives = ["param", "perplexity"]
            idx_to_keep = [all_objectives.index(o) for o in self.objectives]
            all_res = torch.tensor([param, best_perplexity])[idx_to_keep]
            self.logger(f"Config = {config_dict}. ID = {config_id}. Result: #params = {param}, Best perplexity = {best_perplexity}")
            self.logger(f"Adapter params ======== : {params_adapter}, Model params ======== : {params['model']}")
            if self.final_test:
                return all_res
            else:
                return all_res
        else:
            best_acc = -float("inf")  # MCC can be negative; start at -inf so we capture it
            used_test_results = False

            # ---- Try test_results.json first (final_test only) ----
            if test_metrics is not None:
                if "test_matthews_correlation" in test_metrics:
                    best_acc = test_metrics["test_matthews_correlation"]
                    used_test_results = True
                elif "test_spearmanr" in test_metrics:
                    best_acc = test_metrics["test_spearmanr"]
                    used_test_results = True
                elif "test_accuracy" in test_metrics:
                    best_acc = test_metrics["test_accuracy"]
                    used_test_results = True

                if used_test_results and "test_f1" in test_metrics:
                    best_f1 = test_metrics["test_f1"]

                if used_test_results:
                    self.logger(
                        f"Config ID = {config_id}: using test_results.json "
                        f"for final-test metrics (best_acc={best_acc}, "
                        f"best_f1={best_f1})."
                    )
                else:
                    self.logger(
                        f"Config ID = {config_id}: test_results.json present "
                        f"but had no usable metric for task "
                        f"'{self.task_name}'; falling back to "
                        f"trainer_state.json eval metrics."
                    )

            # ---- Fall back to best validation metric from trainer_state.json ----
            if not used_test_results:
                if find_best:
                    for metric in metrics:
                        if "eval_accuracy" in metric and metric["eval_accuracy"] > best_acc:
                            best_acc = metric["eval_accuracy"]
                            if self.task_name == "mrpc":
                                best_f1 = metric.get("eval_f1", best_f1)
                        if metric.get("eval_matthews_correlation", -float("inf")) > best_acc:
                            best_acc = metric["eval_matthews_correlation"]
                        if metric.get("eval_spearmanr", -float("inf")) > best_acc:
                            best_acc = metric["eval_spearmanr"]
                if best_acc == -float("inf"):
                    # No eval ever ran (or trainer_state has no eval_* entries) — default to 0
                    best_acc = 0.0
            print(f"Best acc: {best_acc}")

            best_acc *= float((-1) ** (self.maximization + 1))
            best_f1 *= float((-1) ** (self.maximization + 1))
            all_objectives = ["param", "acc", "f1"]
            idx_to_keep = [all_objectives.index(o) for o in self.objectives]
            all_res = torch.tensor([param, best_acc, best_f1])[idx_to_keep]
            if self.logger is not None:
                self.logger(f"Config = {config_dict}. ID = {config_id}. "
                            f"Result: #params = {param}, Best acc = {best_acc}. Best F1 = {best_f1}")
            if self.final_test:
                return all_res
            else:
                return all_res + self.noise_std * torch.randn_like(all_res)

    #--------------------------------------------------------------------
    # Evaluate Multiple Configurations
    #--------------------------------------------------------------------
    def evaluate_true(self, X: list) -> torch.Tensor:
        if self.mock_run:
            raise NotImplementedError("Mock run not supported in Random Search")
        f = self._evaluate_single
        if self.final_test:
            return f(X)
        res = torch.stack([f(x) for x in X]).view(len(X), self.num_objectives)
        return res

    #--------------------------------------------------------------------
    # Callable Interface
    #--------------------------------------------------------------------
    def __call__(self, X: list) -> torch.Tensor:
        return self.evaluate_true(X)