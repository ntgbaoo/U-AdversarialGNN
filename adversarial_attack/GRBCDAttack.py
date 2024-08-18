from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from torch_geometric.data import Data


from torch_geometric.utils import coalesce, to_undirected
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adversarial_attack.PRBCDAttack import PRBCDAttack


# (predictions, labels, ids/mask) -> Tensor with one element
LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]

class GRBCDAttack(PRBCDAttack):
    r"""The Greedy Randomized Block Coordinate Descent (GRBCD) adversarial
    attack from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    GRBCD shares most of the properties and requirements with
    :class:`PRBCDAttack`. It also uses an efficient gradient based approach.
    However, it greedily flips edges based on the gradient towards the
    adjacency matrix.

    .. note::
        For examples of using the GRBCD Attack, see
        `examples/contrib/rbcd_attack.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/rbcd_attack.py>`_
        for a test time attack (evasion).

    Args:
        model (torch.nn.Module): The GNN module to assess.
        block_size (int): Number of randomly selected elements in the
            adjacency matrix to consider.
        epochs (int, optional): Number of epochs (aborts early if
            :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
        loss (str or callable, optional): A loss to quantify the "strength" of
            an attack. Note that this function must match the output format of
            :attr:`model`. By default, it is assumed that the task is
            classification and that the model returns raw predictions (*i.e.*,
            no output activation) or uses :obj:`logsoftmax`. Moreover, and the
            number of predictions should match the number of labels passed to
            :attr:`attack`. Either pass Callable or one of: :obj:`'masked'`,
            :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
            (default: :obj:`'masked'`)
        is_undirected (bool, optional): If :obj:`True` the graph is
            assumed to be undirected. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """
    coeffs = {'max_trials_sampling': 20, 'eps': 1e-7}

    def __init__(
        self,
       data:Data,
       device = "cpu",
       seed=720
    ):
        super().__init__(data,device,seed)
    
    def update_config(self,
        block_size: int,
        epochs: int = 125,
        loss: Optional[Union[str, LOSS_TYPE]] = 'masked',
        is_undirected: bool = True,
        **kwargs): 
        self.block_size = block_size
        self.epochs = epochs

        if isinstance(loss, str):
            if loss == 'masked':
                self.loss = self._masked_cross_entropy
            elif loss == 'margin':
                self.loss = partial(self._margin_loss, reduce='mean')
            elif loss == 'prob_margin':
                self.loss = self._probability_margin_loss
            elif loss == 'tanh_margin':
                self.loss = self._tanh_margin_loss
            else:
                raise ValueError(f'Unknown loss `{loss}`')
        else:
            self.loss = loss

        self.is_undirected = is_undirected

        self.coeffs.update(kwargs)
    @torch.no_grad()
    def _prepare(self, budget: int) -> List[int]:
        """Prepare attack."""
        self.flipped_edges = self.edge_index.new_empty(2, 0).to(self.device)

        # Determine the number of edges to be flipped in each attach step/epoch
        step_size = budget // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(budget % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * budget

        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)

        return steps

    @torch.no_grad()
    def _update(self, step_size: int, gradient: Tensor, *args,
                **kwargs) -> Dict[str, Any]:
        """Update edge weights given gradient."""
        _, topk_edge_index = torch.topk(gradient, step_size)

        flip_edge_index = self.block_edge_index[:, topk_edge_index]
        flip_edge_weight = torch.ones_like(flip_edge_index[0],
                                           dtype=torch.float32)

        self.flipped_edges = torch.cat((self.flipped_edges, flip_edge_index),
                                       axis=-1)

        if self.is_undirected:
            flip_edge_index, flip_edge_weight = to_undirected(
                flip_edge_index, flip_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')
        edge_index = torch.cat(
            (self.edge_index.to(self.device), flip_edge_index.to(self.device)),
            dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device),
                                 flip_edge_weight.to(self.device)))
        edge_index, edge_weight = coalesce(edge_index, edge_weight,
                                           num_nodes=self.num_nodes,
                                           reduce='sum')

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        # self.edge_weight = torch.ones_like(self.edge_weight)
        assert self.edge_index.size(1) == self.edge_weight.size(0)

        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)

        # Return debug information
        scalars = {
            'number_positive_entries_in_gradient': (gradient > 0).sum().item()
        }
        return scalars

    def _close(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        return self.edge_index, self.flipped_edges
