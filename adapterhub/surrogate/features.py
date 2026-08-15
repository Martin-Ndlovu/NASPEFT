import numpy as np

def compute_num_active_layers(z, num_layers):

    leave_out = z[-num_layers:]

    return num_layers - np.sum(leave_out)


def compute_layer_importance(z, importance_scores, num_layers):

    leave_out = z[-num_layers:]

    active_mask = 1 - leave_out

    score = np.sum(active_mask * importance_scores)

    return score

def build_surrogate_features(z, importance_scores, num_layers):

    z = np.array(z)

    num_active = compute_num_active_layers(z, num_layers)

    importance = compute_layer_importance(
        z, importance_scores, num_layers
    )

    return np.concatenate([
        z,
        [num_active],
        [importance]
    ])