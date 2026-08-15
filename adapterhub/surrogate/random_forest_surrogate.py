import json
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# import root directory from defition in run_naspeft.py
from definition import ROOT_DIR

class RandomForestSurrogate:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        print(f"Predicting with surrogate for {len(X)} candidates...")
        print(f"Input feature shape: {X.shape}")
        print(f"Sample input features (First 5 or available):\n{X[:5]}")

        preds = self.model.predict(X)

        # estimate uncertainty using variance across trees
        tree_preds = np.stack([
            t.predict(X) for t in self.model.estimators_
        ])

        uncert = tree_preds.std(axis=0)

        return preds, uncert


# --------------------------------------------------
# Architecture Feature Utilities
# --------------------------------------------------
def count_active_layers(layer_mask):
    """
    Count number of active layers from binary mask
    """
    return np.sum(layer_mask)


def build_features(Z, derived_features=None):
    """
    Combine search space vector Z with derived features
    """

    if derived_features is None:
        return Z

    return np.concatenate([Z, derived_features], axis=1)


# --------------------------------------------------
# Layer Sensitivity Utilities
# --------------------------------------------------

def load_layer_sensitivity(model_name, task_name, json_path="adapterhub/surrogate/layer_sensitivity.json"):
    """
    Load layer sensitivity scores from json.

    The stored vector MUST be full physical depth (e.g. 12 entries for
    BERT-base), keyed [model_name][task_name]. It holds per_layer_raw_metric
    from the §3.3 probe (accuracy for GLUE, perplexity for causal LM) — i.e.
    the per-layer probe PERFORMANCE, used directly as the importance weight
    (see build_derived_features). run_naspeft.py auto-writes this file from the
    layer-selection summary so it always matches the probe that produced G1.
    """
    # Join root directory with path to json file
    real_json_path = os.path.join(ROOT_DIR, json_path)

    if "/" in model_name:
        model_name = model_name.split("/")[-2] if model_name.endswith("/") else model_name.split("/")[-1]

    with open(real_json_path, "r") as f:
        data = json.load(f)

    return np.array(data[model_name][task_name])


def compute_sensitivity_feature(mask, sensitivity_scores):
    """
    Compute weighted sensitivity score for active layers.

    `mask` here is the ACTIVE mask (1 = active), already inverted from
    leave_out and already sliced to the physical layer count.
    """
    return float(np.sum(mask * sensitivity_scores[:len(mask)]))


# --------------------------------------------------
# Synthetic Architecture Generator (for testing)
# --------------------------------------------------

def generate_architecture_data(n_samples, n_layers=12, n_hparams=3):

    layer_masks = np.random.randint(0, 2, size=(n_samples, n_layers))
    hyperparams = np.random.rand(n_samples, n_hparams)

    Z = np.concatenate([layer_masks, hyperparams], axis=1)

    return layer_masks, hyperparams, Z


def build_derived_features(feature_arrays, sensitivity_scores, num_layers=None):
    """Two derived features per configuration, as the paper's SMBO section
    specifies: the active-layer count and a sensitivity-weighted importance
    score that summarises how much of the high-sensitivity layer budget the
    configuration uses.

    Input encoding (from AdapterSearchSpace.encode_config):
        feature_array = [ physical_leave_out_mask (num_layers) ,
                          log2_reduction_parallel, log2_reduction_prefix,
                          lora_r, lora_alpha ]
    The leave_out mask is at the FRONT (1 = layer frozen, 0 = layer active).

    Feature 1 — active-layer fraction:
        active_i = 1 - leave_out_i          (invert: 1 = active)
        f1 = (sum active_i) / num_layers    (in [0, 1])

    Feature 2 — sensitivity-weighted importance:
        importance_i = per_layer probe performance for layer i
                       (per_layer_raw_metric: accuracy for GLUE, perplexity
                        for causal LM) — positive and monotone in both
                        domains, so no sign issues across tasks.
        f2 = sum_{i active} importance_i = sum( active_i * sensitivity_i )

    Both the mask slice and the importance vector use the FULL physical layer
    count; len(sensitivity_scores) must equal num_layers.
    """
    features = []

    if num_layers is None:
        num_layers = len(sensitivity_scores)

    sens = np.asarray(sensitivity_scores, dtype=float)

    for feature_array in feature_arrays:
        fa = np.asarray(feature_array, dtype=float)

        leave_out = fa[:num_layers]          # FRONT slice (matches encode_config)
        active = 1.0 - leave_out             # invert: 1 = active layer

        active_fraction = float(np.sum(active)) / float(num_layers)
        importance = float(np.sum(active * sens[:num_layers]))

        features.append([active_fraction, importance])

    return np.array(features)


# --------------------------------------------------
# Create Temporary Sensitivity JSON
# --------------------------------------------------

def create_temp_sensitivity_json(path="layer_sensitivity.json"):

    data = {

        "roberta-base": {
            "mrpc": list(np.random.rand(12)),
            "cola": list(np.random.rand(12)),
            "sst2": list(np.random.rand(12))
        },

        "roberta-large": {
            "mrpc": list(np.random.rand(24)),
            "cola": list(np.random.rand(24)),
            "sst2": list(np.random.rand(24))
        }

    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# --------------------------------------------------
# Synthetic Objective Function
# --------------------------------------------------

def generate_fake_objective(layer_masks, hyperparams):

    active_layers = layer_masks.sum(axis=1)

    acc = (
        0.4 * active_layers +
        2 * hyperparams[:, 0] +
        np.random.normal(0, 0.3, len(layer_masks))
    )

    return acc


# --------------------------------------------------
# MAIN TEST
# --------------------------------------------------

def main():

    np.random.seed(42)

    if not os.path.exists("layer_sensitivity.json"):
        print("Creating temporary layer sensitivity JSON")
        create_temp_sensitivity_json()

    model_name = "roberta-base"
    task_name = "mrpc"

    sensitivity_scores = load_layer_sensitivity(model_name, task_name)

    print("Loaded sensitivity scores:", sensitivity_scores)

    N_train = 50
    N_test = 10

    layer_masks, hyperparams, Z = generate_architecture_data(N_train)

    print("Generated training data with shape:", Z.shape)

    derived = build_derived_features(Z, sensitivity_scores, num_layers=12)

    print("Derived features shape:", derived.shape)
    print("Sample derived features:", derived[:5])

    X_train = build_features(Z, derived)

    print("Final training feature shape:", X_train.shape)

    y_train = generate_fake_objective(layer_masks, hyperparams)

    print("Training surrogate...")

    surrogate = RandomForestSurrogate()
    surrogate.fit(X_train, y_train)

    layer_masks_test, hyperparams_test, Z_test = generate_architecture_data(N_test)
    derived_test = build_derived_features(Z_test, sensitivity_scores, num_layers=12)
    X_test = build_features(Z_test, derived_test)

    print("Predicting candidates...")
    pred, uncert = surrogate.predict(X_test)

    for i in range(N_test):
        print(f"Candidate {i} | pred={pred[i]:.3f} | uncert={uncert[i]:.3f}")


if __name__ == "__main__":
    main()