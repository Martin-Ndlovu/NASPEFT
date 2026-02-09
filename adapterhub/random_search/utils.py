#--------------------------------------------------------------------------------------------
# THIS FILE COMPUTES PARETO FRONTS FOR RANDOM SEARCH RESULTS
#--------------------------------------------------------------------------------------------
import torch

def get_pareto_points(Y):
    """
    Compute Pareto front for objectives (minimize Y[:, 0], minimize Y[:, 1]).
    
    Args:
        Y (torch.Tensor): Objective values, shape (N, 2), where Y[:, 0] is param, Y[:, 1] is perplexity.
    
    Returns:
        torch.Tensor: Boolean mask indicating Pareto-optimal points.
    """
    Y = torch.tensor(Y, dtype=torch.float32)  # Minimize param, minimize perplexity
    N = Y.shape[0]
    pareto_mask = torch.ones(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and (Y[j] <= Y[i]).all() and (Y[j] < Y[i]).any():
                pareto_mask[i] = False
    return pareto_mask