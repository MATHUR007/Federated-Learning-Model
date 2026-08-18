# server/fedavg.py

import copy
import torch


def weighted_average_weights(client_states, client_sizes):
    total_samples = sum(client_sizes)
    avg_state = copy.deepcopy(client_states[0])

    for key in avg_state.keys():
        avg_state[key] = avg_state[key] * (client_sizes[0] / total_samples)

    for i in range(1, len(client_states)):
        for key in avg_state.keys():
            avg_state[key] += client_states[i][key] * (client_sizes[i] / total_samples)

    return avg_state