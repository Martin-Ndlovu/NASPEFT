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
import torch
from definition import ROOT_DIR
from accelerate import Accelerator, InitProcessGroupKwargs

accelerator = Accelerator()
# init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
# accelerator = Accelerator(init_process_group_kwargs=init_process_group_kwargs)

# Fallback local settings for WikiText
TASK_SETTINGS = {
    "wikitext": {
        "num_epochs": 3,
        "patience": 3,
        "save_type": "steps",
        "logging_steps": 100,
        "num_steps_per_save": 500,
        "ref_point": [10.0, 100.0],  # [param %, perplexity]
        "objectives": ["param", "perplexity"],
        "low_resource": None,
    }
}
TASK_SETTINGS_SO = {
    "wikitext": {
        "num_epochs": 3,
        "patience": 3,
        "save_type": "steps",
        "logging_steps": 100,
        "num_steps_per_save": 1000,
        "ref_point": [100.0],  # [perplexity]
        "objectives": ["perplexity"],
        "low_resource": None,
    }
}

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
            maximization: bool = False,  # Minimize perplexity
            logger=None,
            seed: int = None,
            resplit_dataset: bool = False,
            final_test: bool = False,
            mock_run: bool = False) -> None:
        self.adapter_name = adapter_name
        self.task_name = task_name
        self.final_test = final_test
        self.objectives = objectives or TASK_SETTINGS[self.task_name].get("objectives", None)
        if len(self.objectives) == 1:
            TASK_SETTINGS[self.task_name]['ref_point'] = TASK_SETTINGS_SO[self.task_name]['ref_point']
            TASK_SETTINGS[self.task_name]['objectives'] = TASK_SETTINGS_SO[self.task_name]['objectives']
            if accelerator.is_main_process:  
                logger.info(f"Task settings: {TASK_SETTINGS[self.task_name]}")
        if accelerator.is_main_process:
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
        self.mock_run = mock_run
        self.resplit_dataset = resplit_dataset
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

        commands = f"""
        cd {ROOT_DIR}/adapterhub
        accelerate launch --main_process_port 0 nas_search.py\
        --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
        --do_train --do_eval --max_seq_length=512 --per_device_train_batch_size=4\
        --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
        --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={logging_steps}\
        --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name}
        """
        commands_resplit = f"""
        cd {ROOT_DIR}/adapterhub
        accelerate launch --main_process_port 0 nas_search.py\
        --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
        --do_train --do_eval --do_predict --resplit_dataset --max_seq_length=512 --per_device_train_batch_size=4\
        --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
        --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={logging_steps}\
        --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name}
        """
        commands_low_resource = f"""
        cd {ROOT_DIR}/adapterhub
        accelerate launch --main_process_port 0 nas_search.py\
        --local_dataset_path={self.data_path} --nas_adapter_config_path={save_file} --model_name_or_path={self.model_path} --task_name={self.task_name}\
        --do_train --do_eval --max_seq_length=256 --per_device_train_batch_size=2\
        --learning_rate=1e-4 --num_train_epochs={num_epochs} --output_dir={pass_output_dir} --patience={patience}\
        --seed={self.seed} --logging_strategy={save_type} --save_strategy={save_type} --logging_steps={logging_steps}\
        --save_steps={num_steps_per_save} --overwrite_output_dir --train_adapter --adapter_name={self.adapter_name} --max_train_samples={low_resource}
        """
        if low_resource is not None:
            subprocess.call(commands_low_resource, shell=True)
        elif not self.resplit_dataset:
            subprocess.call(commands, shell=True)
        else:
            subprocess.call(commands_resplit, shell=True)

        if accelerator.is_main_process:
            trainer_state = json.load(
                open(os.path.join(save_path_this_config, "trainer_state.json")))
            metrics = trainer_state["log_history"]
            params = json.load(
                open(os.path.join(save_path_this_config, "model_param_dict.json")))
            params_adapter = params["adapters"]

            # Calculation of the number of adapter parameters based on config
            # max_layer = 16# Llama-3.1-1B
            # hidden_size = 2048 # Llama-3.1-1B
            # if self.adapter_name == "naspeft":
            #     num_adapter_layers = 0
            #     for i in range(max_layer):
            #         if not config_dict.get(f"leave_out_{i}", False):
            #             num_adapter_layers += 1
            #     self.logger(f"Number of layers with adapters: {num_adapter_layers}")
            #     reduction_prefix = hidden_size / config_dict['reduction_prefix']
            #     bottleneck_size = hidden_size / config_dict['reduction_parallel']
            #     lora_r = 64 / config_dict.get('lora_r', 16) 
            #     if config_dict['reduction_prefix'] == hidden_size:
            #         reduction_prefix = 1
            #     if config_dict['reduction_parallel'] == hidden_size:
            #         bottleneck_size = 1
            #     if config_dict.get('lora_r', 16) == 64:
            #         lora_r = 1
            #     prefix_size = reduction_prefix * hidden_size * num_adapter_layers * 2
            #     parallel_size = bottleneck_size * 2 * hidden_size * num_adapter_layers
            #     lora_size = 2 * lora_r * hidden_size * num_adapter_layers * 4  # 4 target modules
            #     if config_dict['reduction_prefix'] > hidden_size:
            #         prefix_size = 0
            #     if config_dict['reduction_parallel'] > hidden_size:
            #         parallel_size = 0
            #     if config_dict.get('lora_r', 16) > hidden_size:
            #         lora_size = 0
            #     params_adapter = prefix_size + parallel_size + lora_size
            
            # param = float(params_adapter) / float(params['model']) * 100. * float(((-1) ** self.maximization))


            # Ignore the above calculations and use the true number of parameters instead
            param = float(params_adapter) / float(params['model']) * 100. * float(((-1) ** self.maximization))

            best_perplexity = float('inf')
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

    def evaluate_true(self, X: list) -> torch.Tensor:
        if self.mock_run:
            raise NotImplementedError("Mock run not supported in Random Search")
        f = self._evaluate_single
        if self.final_test:
            return f(X)
        res = torch.stack([f(x) for x in X]).view(len(X), self.num_objectives)
        return res

    def __call__(self, X: list) -> torch.Tensor:
        return self.evaluate_true(X)