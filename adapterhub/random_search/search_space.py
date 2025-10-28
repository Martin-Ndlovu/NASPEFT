import copy
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from collections import OrderedDict

class AdapterSearchSpace:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.cs = None
        self.create_search_space()
        self.dim = len(self.cs)

    def create_search_space(self):
        self.cs = CS.ConfigurationSpace(seed=self.seed)
        params = [
            CSH.UniformIntegerHyperparameter(
                "log2_reduction_parallel", 0, 11, default_value=1),
            CSH.UniformIntegerHyperparameter(
                "log2_reduction_prefix", 0, 11, default_value=1),
            CSH.CategoricalHyperparameter(
                "lora_r", [8, 16, 32, 64], default_value=16),
            CSH.CategoricalHyperparameter(
                "lora_alpha", [8, 16, 32], default_value=16),
        ]
        # Add leave_out parameters for Llama’s 16 layers
        for i in range(16):
            params.append(
                CSH.UniformIntegerHyperparameter(
                    f"leave_out_{i}", 0, 1, default_value=0)
            )
        self.cs.add(params)

    def encode_config(self, config: CS.Configuration, use_orig: bool = False) -> np.ndarray:
        orig_array_rep = config.get_array()
        if use_orig:
            return orig_array_rep
        return orig_array_rep

    def config_to_dict(self, config: CS.Configuration) -> Dict[str, Any]:
        ret = {k: int(v) for k, v in config.get_dictionary().items()}
        ret["reduction_parallel"] = 2 ** ret["log2_reduction_parallel"]
        del ret["log2_reduction_parallel"]
        ret["reduction_prefix"] = 2 ** ret["log2_reduction_prefix"]
        del ret["log2_reduction_prefix"]
        # lora_r and lora_alpha are used directly (no transformation)
        return ret

    def dict_to_config(self, config_dict: Dict[str, Any]) -> CS.Configuration:
        config_dict = copy.deepcopy(config_dict)
        config_dict["log2_reduction_parallel"] = int(np.log2(config_dict["reduction_parallel"]))
        del config_dict["reduction_parallel"]
        config_dict["log2_reduction_prefix"] = int(np.log2(config_dict["reduction_prefix"]))
        del config_dict["reduction_prefix"]
        # lora_r and lora_alpha remain unchanged
        return CS.Configuration(self.cs, config_dict)

    def sample_configuration(self, return_dict_repr: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        is_valid = False
        while not is_valid:
            config = self.cs.sample_configuration()
            is_valid = self.is_valid(config)
        array_rep = self.encode_config(config)
        dict_rep = self.config_to_dict(config)
        if return_dict_repr:
            return array_rep, dict_rep
        return array_rep

    def get_config_id(self, config: Union[CS.Configuration, Dict[str, Any]]) -> str:
        if isinstance(config, CS.Configuration):
            config = self.config_to_dict(config)

        def sort_key(item):
            key, _ = item
            if key.startswith("leave_out_"):
                return (0, int(key.split("_")[-1]))  # numeric order
            return (1, key)  # keep other params after leave_out_*
        
        config = OrderedDict(sorted(config.items(), key=sort_key))  # Ensure consistent ordering
        str_id = "_".join([str(i) for i in config.values()])

        return str_id

    def is_valid(self, config: Union[CS.Configuration, Dict[str, Any]]):
        max_layer = 16  # Llama 3.1 8B has 32 layers
        num_layer = 16
        for i in range(max_layer):
            if config[f"leave_out_{i}"]:
                num_layer -= 1
        if num_layer == 0:
            return False
        # Ensure at least one adapter is effective
        return (
            config['log2_reduction_parallel'] < 11 or
            config['log2_reduction_prefix'] < 11 or
            config['lora_r'] <= 64  # lora_r and lora_alpha are constrained by categorical choices
        )