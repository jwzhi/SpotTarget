import dgl
import torch
from ogb.linkproppred import DglLinkPropPredDataset
import copy


def split_check(g, edge_split, valid_as_input=False):
    """Find check if graph g has message passing leakages and generate the proper inference graph according 
    to user requirement
    Parameters
    ----------
    g : DGLGraph
        The inference graph.
    edge_split : dict[key, Tensor]
        The dict needs to contain the following keys:
            'valid': m*2 tensor
                where m is the number of validation edges.
            'test': n*2 tensor
                where n is the number of test edges. 
    valid_as_input: True or False
        Whether to use validation edges as input edges in g. 
        If set as True, then validation edges will be added into g if it does not exist. 
        If set as False, then validation 

    Examples
    ----------    
    dataset = DglLinkPropPredDataset('ogbl-collab')
    g = dataset[0]
    edge_split = dataset.get_edge_split()
    edge_split_new = {}
    edge_split_new['valid'] = edge_split['valid']['edge']
    edge_split_new['test'] = edge_split['test']['edge']
    inference_g = split_check(g, edge_split_new)
    """
    valid_edges = edge_split['valid']
    test_edges = edge_split['test']
    inference_g = copy.deepcopy(g)
    has_valid_edges = g.has_edges_between(valid_edges[:, 0], valid_edges[:, 1])
    has_test_edges = g.has_edges_between(test_edges[:, 0], test_edges[:, 1])
    edges_to_remove = []
    if not valid_as_input:
        if has_valid_edges.float().mean().item() > 0:
            for s, t in valid_edges[has_valid_edges]:
                if g.has_edges_between(s, t):
                    edges_to_remove.append(g.edge_ids(s, t, return_uv=True)[-1])
                if g.has_edges_between(t, s):
                    edges_to_remove.append(g.edge_ids(t, s, return_uv=True)[-1])

    if has_test_edges.float().mean().item() > 0:
        for s, t in test_edges[has_test_edges]:
            if g.has_edges_between(s, t):
                edges_to_remove.append(g.edge_ids(s, t, return_uv=True)[-1])
            if g.has_edges_between(t, s):
                edges_to_remove.append(g.edge_ids(t, s, return_uv=True)[-1])
    edges_to_remove = [item for sublist in edges_to_remove for item in sublist]
    inference_g.remove_edges(edges_to_remove)
    return inference_g


dataset = DglLinkPropPredDataset('ogbl-collab')
g = dataset[0]
edge_split = dataset.get_edge_split()
edge_split_new = {}
edge_split_new['valid'] = edge_split['valid']['edge']
edge_split_new['test'] = edge_split['test']['edge']
inference_g = split_check(g, edge_split_new)
has_valid_edges = inference_g.has_edges_between(edge_split['valid']['edge'][:, 0], edge_split['valid']['edge'][:, 1])
has_test_edges = inference_g.has_edges_between(edge_split['test']['edge'][:, 0], edge_split['test']['edge'][:, 1])
print(has_valid_edges.float().mean().item())
print(has_test_edges.float().mean().item())
