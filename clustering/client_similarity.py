# clustering/client_similarity.py

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def flatten_state_dict(state_dict):
    flat_params = []
    for key in state_dict.keys():
        flat_params.append(state_dict[key].detach().cpu().numpy().ravel())
    return np.concatenate(flat_params)


def get_client_update(global_state, local_state):
    update = {}
    for key in global_state.keys():
        update[key] = local_state[key] - global_state[key]
    return update


def build_similarity_matrix(global_state, client_states, client_ids):
    client_vectors = []

    for state in client_states:
        update = get_client_update(global_state, state)
        flat_update = flatten_state_dict(update)
        client_vectors.append(flat_update)

    sim_matrix = cosine_similarity(client_vectors)
    return sim_matrix, client_ids, client_vectors