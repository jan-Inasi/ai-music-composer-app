import torch
import itertools as it
import muspy as mp
import numpy as np
import lib.continous_pd_representation as repr_cpd


def repetitiveness(tunes: torch.Tensor | list[torch.Tensor]) -> float:
    """measures repetitions in a batch of tunes"""
    if isinstance(tunes, torch.Tensor):
        length = tunes.shape[2]
        tune_count = tunes.shape[0]
    else:
        length = tunes[0].shape[1]
        tune_count = len(tunes)
    combination_count = tune_count * (tune_count - 1) // 2

    same_cell_counter = 0
    for tune_1, tune_2 in it.combinations(tunes, 2):
        same_cell_counter += (tune_1 == tune_2).int().sum().item()

    return same_cell_counter / length / combination_count


def scale_consistency(tunes):
    tunes = [repr_cpd.pd2muspy(tune.T) for tune in tunes]
    values = np.array([mp.scale_consistency(t) for t in tunes])
    return (np.mean(values), np.var(values), np.min(values), np.max(values))


def pitch_range(tunes):
    tunes = [repr_cpd.pd2muspy(tune.T) for tune in tunes]
    values = np.array([mp.pitch_range(t) for t in tunes])
    return np.mean(values)
