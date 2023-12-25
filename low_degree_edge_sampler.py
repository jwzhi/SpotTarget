from dgl.dataloading.base import EdgePredictionSampler, _find_exclude_eids
from collections.abc import Mapping
from dgl.base import EID, NID
from dgl.utils import recursive_apply
import dgl
import torch


def find_exclude_eids(g, seed_edges, exclude, reverse_eids=None, reverse_etypes=None,
                      output_device=None, degree_threshold=10):
    """Find all edge IDs to exclude according to :attr:`exclude_mode`.
    Parameters
    ----------
    g : DGLGraph
        The graph.
    exclude_mode : str, optional
        Can be either of the following,
        None (default)
            Does not exclude any edge.
        'self'
            Exclude the given edges themselves but nothing else.
        'reverse_id'
            Exclude all edges specified in ``eids``, as well as their reverse edges
            of the same edge type.
            The mapping from each edge ID to its reverse edge ID is specified in
            the keyword argument ``reverse_eid_map``.
            This mode assumes that the reverse of an edge with ID ``e`` and type
            ``etype`` will have ID ``reverse_eid_map[e]`` and type ``etype``.
        'reverse_types'
            Exclude all edges specified in ``eids``, as well as their reverse
            edges of the corresponding edge types.
            The mapping from each edge type to its reverse edge type is specified
            in the keyword argument ``reverse_etype_map``.
            This mode assumes that the reverse of an edge with ID ``e`` and type ``etype``
            will have ID ``e`` and type ``reverse_etype_map[etype]``.
        callable
            Any function that takes in a single argument :attr:`seed_edges` and returns
            a tensor or dict of tensors.
    eids : Tensor or dict[etype, Tensor]
        The edge IDs.
    reverse_eids : Tensor or dict[etype, Tensor]
        The mapping from edge ID to its reverse edge ID.
    reverse_etypes : dict[etype, etype]
        The mapping from edge etype to its reverse edge type.
    output_device : device
        The device of the output edge IDs.
    """
    src, dst = g.find_edges(seed_edges)
    head_degree = g.in_degrees(src)
    tail_degree = g.in_degrees(dst)
    degree = torch.min(head_degree, tail_degree)
    degree_mask = degree < degree_threshold
    edges_need_to_exclude = seed_edges[degree_mask]
    exclude_eids = _find_exclude_eids(
        g,
        exclude,
        edges_need_to_exclude,
        reverse_eid_map=reverse_eids,
        reverse_etype_map=reverse_etypes)
    if exclude_eids is not None and output_device is not None:
        exclude_eids = recursive_apply(exclude_eids, lambda x: F.copy_to(x, output_device))
    return exclude_eids


class EdgePredictionSamplerwithDegree(EdgePredictionSampler):
    """Sampler class that builds upon EdgePredictionSampler
    The exlucde train target is only done on edges with a degree < degree threshold

    ------------------------------
    Need to call this directly in the code instead of calling as_edge_prediction_sampler
    """

    def __init__(self, sampler, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_labels=None, degree_threshold=10):
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler, prefetch_labels)
        self.degree_threshold = degree_threshold

    def sample(self, g, seed_edges):  # pylint: disable=arguments-differ
        """Samples a list of blocks, as well as a subgraph containing the sampled
        edges from the original graph.
        If :attr:`negative_sampler` is given, also returns another graph containing the
        negative pairs as edges.
        """
        if isinstance(seed_edges, Mapping):
            seed_edges = {g.to_canonical_etype(k): v for k, v in seed_edges.items()}
        exclude = self.exclude
        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device)
        eids = pair_graph.edata[EID]

        if self.negative_sampler is not None:
            neg_graph = self._build_neg_graph(g, seed_edges)
            pair_graph, neg_graph = dgl.compact_graphs([pair_graph, neg_graph])
        else:
            pair_graph = dgl.compact_graphs(pair_graph)

        pair_graph.edata[EID] = eids
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = find_exclude_eids(
            g, seed_edges, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device, self.degree_threshold)

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes, exclude_eids)

        if self.negative_sampler is None:
            return self.assign_lazy_features((input_nodes, pair_graph, blocks))
        else:
            return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))
