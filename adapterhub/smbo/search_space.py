#-----------------------------------------------------------------------------------------
# THIS FILE DEFINES THE SEARCH SPACE FOR SMBO OVER ADAPTER CONFIGURATIONS
#
# Layer-selection integration (paper §3.3):
#   The search space is REDUCED to the |G1| searchable layers chosen by the
#   layer-sensitivity probe. Internally ConfigSpace carries only |G1| mask
#   variables (leave_out_0 .. leave_out_{|G1|-1}); this is the true search-space
#   dimensionality reduction. EVERYTHING this class EMITS, however, is a FULL
#   physical-layer config of width `num_physical_layers` (e.g. 12 for BERT-base):
#
#     - config_to_dict(...)   -> dict with leave_out_0 .. leave_out_{P-1}
#                                (G2 layers forced leave_out=1, G1 bits placed
#                                 at their physical positions)
#     - encode_config(...)    -> np.ndarray whose first P entries are the
#                                physical leave_out mask, followed by the
#                                module hyperparameters
#
#   This makes the reduction invisible to every caller (Problem /
#   nas_search_plus.py / param_oracle / build_derived_features / surrogate):
#   they all receive consistent P-dim physical configs and need no changes.
#
#   G1 is passed EXPLICITLY into the constructor. No file auto-discovery.
#-----------------------------------------------------------------------------------------
import copy
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, List, Sequence
from collections import OrderedDict
import json
import os


class AdapterSearchSpace:
    #-------------------------------------------------------------------------------------
    # Initialize Search Space
    #
    #   g1                  : explicit list of PHYSICAL layer indices that are
    #                         searchable (the searchable_layers / G1 produced by
    #                         layer_selection.py). If None, the full physical
    #                         layer range is searchable (no reduction).
    #   num_physical_layers : total transformer layers in the backbone (e.g. 12
    #                         for BERT-base, 24 for -large, 16 for LLaMA). If
    #                         None, inferred from is_large / is_llama defaults.
    #-------------------------------------------------------------------------------------
    def __init__(self,
                 seed: Optional[int] = None,
                 is_large: bool = False,
                 is_llama: bool = False,
                 is_t5: bool = False,
                 g1: Optional[Sequence[int]] = None,
                 num_physical_layers: Optional[int] = None):
        self.seed = seed
        self.is_large = is_large
        self.is_llama = is_llama
        self.is_t5 = is_t5

        # ---- Resolve the physical layer count (the width of every EMITTED config) ----
        if num_physical_layers is not None:
            self.num_physical_layers = int(num_physical_layers)
        elif self.is_llama:
            self.num_physical_layers = 16
        else:
            self.num_physical_layers = 24 if self.is_large else 12

        # ---- Resolve G1 (searchable physical layer indices) ----
        if g1 is None:
            # No reduction: every physical layer is searchable.
            self.g1 = list(range(self.num_physical_layers))
        else:
            self.g1 = sorted(int(i) for i in g1)
            for i in self.g1:
                if not (0 <= i < self.num_physical_layers):
                    raise ValueError(
                        f"G1 index {i} out of range for "
                        f"num_physical_layers={self.num_physical_layers}"
                    )
        if len(self.g1) == 0:
            raise ValueError("G1 is empty — nothing to search over.")

        # G2 = physical layers NOT in G1 -> always frozen (leave_out=1).
        self.g2 = [i for i in range(self.num_physical_layers) if i not in set(self.g1)]

        # The ConfigSpace search dimension over layers is |G1|, NOT the full
        # physical count. This is the search-space reduction from §3.3.
        self.num_search_layers = len(self.g1)

        self.cs = None
        self.create_search_space()
        self.dim = len(self.cs)

    #-------------------------------------------------------------------------------------
    # Create Search Space
    #-------------------------------------------------------------------------------------
    def create_search_space(self):
        self.cs = CS.ConfigurationSpace(seed=self.seed)

        if self.is_llama:
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
        elif self.is_t5:
            max_log2 = 10 if self.is_large else 9
            params = [
                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_parallel", 0, max_log2, default_value=1),
                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_prefix", 0, max_log2, default_value=1),
                CSH.CategoricalHyperparameter(
                    "lora_r", [8, 16, 32, 64], default_value=16),
                CSH.CategoricalHyperparameter(
                    "lora_alpha", [8, 16, 32], default_value=16),
            ]
        else:
            params = [
                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_parallel", 0, 10, default_value=1),
                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_prefix", 0, 10, default_value=1),
                CSH.CategoricalHyperparameter(
                    "lora_r", [8, 16, 32, 64], default_value=16),
                CSH.CategoricalHyperparameter(
                    "lora_alpha", [8, 16, 32], default_value=16),
            ]
        #--------------------------------------------------------------------------------
        # Add leave_out parameters: ONE PER SEARCHABLE (G1) LAYER, not per
        # physical layer. These are indexed 0 .. |G1|-1 in SEARCH space and are
        # remapped to physical positions on emission.
        #--------------------------------------------------------------------------------
        for k in range(self.num_search_layers):
            params.append(
                CSH.UniformIntegerHyperparameter(
                    f"leave_out_{k}", 0, 1, default_value=0)
            )
        self.cs.add(params)

    #-------------------------------------------------------------------------------------
    # Detect number of layers — kept for backward compatibility.
    #   Returns the PHYSICAL layer count (the width of emitted configs).
    #-------------------------------------------------------------------------------------
    def _get_num_layers(self, json_path: Optional[str] = None):
        return self.num_physical_layers

    #-------------------------------------------------------------------------------------
    # Core remap: SEARCH-space leave_out (|G1| bits) -> PHYSICAL leave_out (P bits)
    #
    #   G2 layers are ALWAYS frozen (leave_out=1). For each G1 layer at search
    #   index k, the sampled bit leave_out_k is written to its physical
    #   position self.g1[k]. This single helper is used by BOTH config_to_dict
    #   and encode_config so the two representations can never diverge.
    #-------------------------------------------------------------------------------------
    def _search_mask_to_physical(self, search_leave_out: Dict[int, int]) -> List[int]:
        physical = [1] * self.num_physical_layers  # default: frozen
        for k, phys_idx in enumerate(self.g1):
            physical[phys_idx] = int(search_leave_out.get(k, 0))
        # G2 positions stay 1 (frozen) by construction.
        return physical

    #-------------------------------------------------------------------------------------
    # Configuration Encoding
    #
    #   Returns a PHYSICAL-width vector:
    #       [ physical_leave_out_0 .. physical_leave_out_{P-1},
    #         <module hyperparameter array entries> ]
    #   so the surrogate's build_derived_features (which slices mask[:num_layers]
    #   with num_layers = physical count) sees the correct physical mask, and
    #   Z stays consistent with the dict from config_to_dict.
    #-------------------------------------------------------------------------------------
    def encode_config(self, config: CS.Configuration, use_orig: bool = False) -> np.ndarray:
        if use_orig:
            return config.get_array()

        d = config.get_dictionary()
        search_leave_out = {
            int(k.split("_")[-1]): int(v)
            for k, v in d.items() if k.startswith("leave_out_")
        }
        physical_mask = self._search_mask_to_physical(search_leave_out)

        # Module hyperparameters in a fixed, stable order.
        hp_order = ["log2_reduction_parallel", "log2_reduction_prefix",
                    "lora_r", "lora_alpha"]
        hp_vals = [float(d[k]) for k in hp_order]

        return np.asarray(
            [float(b) for b in physical_mask] + hp_vals,
            dtype=np.float64,
        )

    #-------------------------------------------------------------------------------------
    # Configuration Conversion -> PHYSICAL-width dict
    #
    #   Emits leave_out_0 .. leave_out_{P-1} (physical), plus the module
    #   hyperparameters. G2 layers are leave_out=1; G1 layers carry the
    #   sampled bit at their physical index. nas_search_plus.py iterates
    #   range(num_hidden_layers) reading leave_out_i — this now matches.
    #-------------------------------------------------------------------------------------
    def config_to_dict(self, config: CS.Configuration) -> Dict[str, Any]:
        raw = {k: int(v) for k, v in config.get_dictionary().items()}

        search_leave_out = {
            int(k.split("_")[-1]): int(v)
            for k, v in raw.items() if k.startswith("leave_out_")
        }
        physical_mask = self._search_mask_to_physical(search_leave_out)

        ret: Dict[str, Any] = {}
        for i in range(self.num_physical_layers):
            ret[f"leave_out_{i}"] = physical_mask[i]

        ret["reduction_parallel"] = 2 ** raw["log2_reduction_parallel"]
        ret["reduction_prefix"] = 2 ** raw["log2_reduction_prefix"]
        ret["lora_r"] = raw["lora_r"]
        ret["lora_alpha"] = raw["lora_alpha"]
        return ret

    #-------------------------------------------------------------------------------------
    # Convert PHYSICAL dict back to a SEARCH-space Configuration
    #
    #   Inverse of config_to_dict. Reads physical leave_out_i for i in G1 and
    #   maps them to search indices 0..|G1|-1. G2 entries are ignored (they are
    #   structurally forced to 1 and are not search variables). Raises if a G2
    #   layer is unexpectedly marked active, which would indicate a corrupt
    #   physical config.
    #-------------------------------------------------------------------------------------
    def dict_to_config(self, config_dict: Dict[str, Any]) -> CS.Configuration:
        cd = copy.deepcopy(config_dict)

        g1_set = set(self.g1)
        for i in self.g2:
            if int(cd.get(f"leave_out_{i}", 1)) == 0:
                raise ValueError(
                    f"Physical layer {i} is in G2 (must be frozen) but the "
                    f"config marks it active (leave_out=0). Corrupt config."
                )

        search_cfg: Dict[str, Any] = {}
        for k, phys_idx in enumerate(self.g1):
            search_cfg[f"leave_out_{k}"] = int(cd.get(f"leave_out_{phys_idx}", 0))

        search_cfg["log2_reduction_parallel"] = int(np.log2(cd["reduction_parallel"]))
        search_cfg["log2_reduction_prefix"] = int(np.log2(cd["reduction_prefix"]))
        search_cfg["lora_r"] = cd["lora_r"]
        search_cfg["lora_alpha"] = cd["lora_alpha"]
        return CS.Configuration(self.cs, search_cfg)

    #-------------------------------------------------------------------------------------
    # Sample Valid Configuration
    #-------------------------------------------------------------------------------------
    def sample_configuration(self, return_dict_repr: bool = False):
        is_valid = False
        while not is_valid:
            config = self.cs.sample_configuration()
            is_valid = self.is_valid(config)
        array_rep = self.encode_config(config)
        dict_rep = self.config_to_dict(config)
        if return_dict_repr:
            return array_rep, dict_rep
        return array_rep

    #-------------------------------------------------------------------------------------
    # Get Configuration ID  (operates on the PHYSICAL dict for stable, unique IDs)
    #-------------------------------------------------------------------------------------
    def get_config_id(self, config: Union[CS.Configuration, Dict[str, Any]]) -> str:
        if isinstance(config, CS.Configuration):
            config = self.config_to_dict(config)

        def sort_key(item):
            key, _ = item
            if key.startswith("leave_out_"):
                return (0, int(key.split("_")[-1]))
            return (1, key)

        config = OrderedDict(sorted(config.items(), key=sort_key))
        return "_".join([str(i) for i in config.values()])

    #-------------------------------------------------------------------------------------
    # Validate Configuration
    #
    #   Validity is checked on the SEARCH-space configuration: at least one G1
    #   layer active, and at least one effective module. (G2 layers are frozen
    #   structurally and do not affect validity.)
    #-------------------------------------------------------------------------------------
    def is_valid(self, config: Union[CS.Configuration, Dict[str, Any]]):
        if isinstance(config, CS.Configuration):
            d = config.get_dictionary()
        else:
            d = config

        active_g1 = 0
        for k in range(self.num_search_layers):
            if not int(d.get(f"leave_out_{k}", 0)):
                active_g1 += 1
        if active_g1 == 0:
            return False

        return (
            d['log2_reduction_parallel'] < 11 or
            d['log2_reduction_prefix'] < 11 or
            d['lora_r'] in [8, 16, 32, 64]
        )