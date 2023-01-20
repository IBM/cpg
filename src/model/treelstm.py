import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import init
import torch.nn.functional as F
from torch.nn import functional as F

from . import basic
import numpy as np

class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_d, out_d))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

class TypePredictor(nn.Module):

    def __init__(self, repr_dim, hidden_dims, num_types, gumbel_temp=1.):
        super().__init__()
        self.net = FeedForward(repr_dim, hidden_dims, num_types)
        self.gumbel_temp = gumbel_temp

    def reduce_gumbel_temp(self, factor, iter, verbose=False):
        new_temp = np.maximum(self.gumbel_temp * np.exp(-factor * iter), 0.5)
        if verbose:
            print(f'Gumbel temp lowered from {self.gumbel_temp:g} to {new_temp:g}')
        self.gumbel_temp = new_temp

    def score(self, x):
        return self.net(x)

    def sample(self, type_scores):
        distribution = F.gumbel_softmax(type_scores, tau=self.gumbel_temp, dim=-1)
        samples = torch.argmax(distribution, dim=-1)
        return samples, distribution

    def forward(self, x):
        type_scores = self.score(x)
        samples, distribution = self.sample(type_scores)
        return samples, distribution


class TypedBinaryTreeLSTMLayer(nn.Module):
    def __init__(self, hidden_value_dim, hidden_type_dim, type_predictor, binary_type_predictor):
        super(TypedBinaryTreeLSTMLayer, self).__init__()
        self.hidden_value_dim = hidden_value_dim
        self.hidden_type_dim = hidden_type_dim
        self.comp_linear_v = nn.Linear(in_features=2 * (hidden_value_dim + hidden_type_dim),
                                       out_features=5 * hidden_value_dim)
        self.type_predictor = type_predictor
        self.binary_type_predictor = binary_type_predictor
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear_v.weight.data)
        #init.orthogonal_(self.comp_linear_t.weight.data)    # TODO: is this what we want?
        init.constant_(self.comp_linear_v.bias.data, val=0)
        #init.constant_(self.comp_linear_t.bias.data, val=0)
        # type predictor's parameter are not reset

    def forward(self, l=None, r=None, positions_force=None, target_types=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_value_dim + hidden_type_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_value_dim + hidden_type_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_value_dim + hidden_type_dim).
        """

        # extract state from left and right args
        hl, cl = l
        hr, cr = r

        # extract value and type information
        hl_v, hl_t = torch.split(hl, [self.hidden_value_dim, self.hidden_type_dim], dim=2)  # hl
        hr_v, hr_t = torch.split(hr, [self.hidden_value_dim, self.hidden_type_dim], dim=2)  # hr

        # compute updated hidden state and memory values
        # TK changed to include type information
        hlr_cat_v = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear_v(hlr_cat_v)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
               + u.tanh() * i.sigmoid())
        h_v = o.sigmoid() * c.tanh()

        if target_types != None:
            _, sem_h_t = self.type_predictor(h_v)
            B, L, T = sem_h_t.shape
            positions = torch.zeros(B).long()
            for i in range(B):
                if positions_force[i] == []:
                    positions[i] = 0
                else:
                    positions[i] = positions_force[i][-1]
            hom_loss = F.cross_entropy(torch.clamp(sem_h_t[torch.arange(B), positions].view(B, T), min=1e-10, max=1.0).log(),
                                       target_types.long())
            new_h = torch.cat([h_v, sem_h_t], dim=2)
        else:
            # compute updated hidden state and memory types
            hlr_cat_t = torch.cat([hl_t, hr_t], dim=2)
            _, h_t = self.binary_type_predictor(hlr_cat_t)

            # compute type prediction from semantic value
            _, sem_h_t = self.type_predictor(h_v)

            # compute homomorphic loss
            kl_loss = torch.nn.KLDivLoss()
            hom_loss = kl_loss(torch.clamp(h_t, min=1e-10, max=1.0).log(), torch.clamp(sem_h_t, min=1e-10, max=1.0))

            # concatenate value and type information for the hidden state
            new_h_type = F.one_hot(torch.argmax((h_t + sem_h_t) / 2.0, dim=-1), num_classes=self.hidden_type_dim)
            new_h = torch.cat([h_v, new_h_type], dim=2)

        return (new_h, c), hom_loss

class TypedBinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_value_dim, hidden_type_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature, bidirectional, is_lstm=False, scan_token_to_type_map=None,
                 input_tokens=None, positions_force=None, types_force=None):
        super(TypedBinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_value_dim = hidden_value_dim
        self.hidden_type_dim = hidden_type_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.gumbel_temperature = gumbel_temperature
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm
        self.scan_token_to_type_map = scan_token_to_type_map
        self.input_tokens = input_tokens
        self.positions_force = positions_force
        self.types_force = types_force

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_value_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_value_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_value_dim)
        self.type_predictor = TypePredictor(hidden_value_dim, [32, 32], hidden_type_dim)
        self.binary_type_predictor = TypePredictor(2 * hidden_type_dim, [32, 32], hidden_type_dim)
        if self.bidirectional:
            self.treelstm_layer = TypedBinaryTreeLSTMLayer(2 * hidden_value_dim, 2 * hidden_type_dim, self.type_predictor, self.binary_type_predictor)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * (hidden_value_dim + hidden_type_dim)))
        else:
            self.treelstm_layer = TypedBinaryTreeLSTMLayer(hidden_value_dim, hidden_type_dim, self.type_predictor, self.binary_type_predictor)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_value_dim + hidden_type_dim))

        self.reset_parameters()

    def reduce_gumbel_temp(self, iter):
        self.type_predictor.reduce_gumbel_temp(0.0003, iter, verbose=True)
        self.binary_type_predictor.reduce_gumbel_temp(0.0003, iter, verbose=True)

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal_(self.word_linear.weight.data)
            init.constant_(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal_(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask, positions_force=None):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = (self.comp_query * new_h).sum(-1)
        comp_weights = comp_weights / math.sqrt(self.hidden_value_dim + self.hidden_type_dim)
        B, L = comp_weights.size()
        if positions_force != None:
            positions = torch.zeros(B).long()
            for i in range(B):
                if positions_force[i] == []:
                    positions[i] = 0
                else:
                    positions[i] = positions_force[i][-1]
            select_mask = F.one_hot(positions, comp_weights.size(1))
            select_mask = select_mask.float()
        elif not self.is_lstm:
            if self.training:
                select_mask = basic.st_gumbel_softmax(
                    logits=comp_weights, temperature=self.gumbel_temperature,
                    mask=mask)
            else:
                select_mask = basic.greedy_select(logits=comp_weights, mask=mask)
                select_mask = select_mask.float()
        else:
            select_mask = F.one_hot(torch.zeros(comp_weights.size(0)).long(), comp_weights.size(1))
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand[:, :, :300] * new_c
                 + left_mask_expand[:, :, :300] * old_c_left
                 + right_mask_expand[:, :, :300] * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, input_tokens, return_select_masks=False, positions_force=None, types_force=None):
        max_depth = input.size(1)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)
        select_masks = []
        hom_loss_sum = 0

        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new_zeros(batch_size, self.hidden_dim)
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state_v = self.word_linear(input)
            h_v, c = state_v.chunk(chunks=2, dim=2)
            h_t_samples, h_t = self.type_predictor(h_v)
            if self.scan_token_to_type_map is not None:
                # compute cross entropy loss of predicted type against the oracle
                B, L, T = h_t.shape
                target_types = self.scan_token_to_type_map[input_tokens.view(B*L)]
                hom_loss_sum = F.cross_entropy(torch.clamp(h_t.view(B*L, T), min=1e-10, max=1.0).log(), target_types)

            h = torch.concat((h_v, F.one_hot(h_t_samples, num_classes=self.hidden_type_dim)), dim=2)
            state = h, c
        nodes = []
        if self.intra_attention:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            B, _, _ = h.size()
            if types_force != None:
                # get target types for this step and remove them
                target_types = torch.zeros(B)
                for k in range(len(types_force)):
                    if types_force[k] != []:
                        target_types[k] = types_force[k][-1]
                        types_force[k].pop(-1)
            else:
                target_types = None
            new_state, hom_loss = self.treelstm_layer(l=l, r=r, positions_force=positions_force, target_types=target_types)
            hom_loss_sum += hom_loss
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:], positions_force=positions_force)
                # remove positions at this step
                if positions_force != None:
                    for k in range(len(positions_force)):
                        if positions_force[k] != []:
                            positions_force[k].pop(-1)
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                if self.intra_attention:
                    nodes.append(selected_h)
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.float()
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = basic.masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
        assert h.size(1) == 1 and c.size(1) == 1
        if not return_select_masks:
            return h.squeeze(1), c.squeeze(1), hom_loss_sum
        else:
            return h.squeeze(1), c.squeeze(1), hom_loss_sum, select_masks
