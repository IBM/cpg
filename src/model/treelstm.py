import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import init
import torch.nn.functional as F
from torch.nn import functional as F

from . import basic
import numpy as np
import math


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

    def reset_parameters(self):
        # TK: TODO: do we have to do anything here?
        pass

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

    def reset_parameters(self):
        self.net.reset_parameters()

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
    def __init__(self, hidden_value_dim, hidden_type_dim, type_predictor, binary_type_predictor,
                 max_seq_len, decoder):
        super(TypedBinaryTreeLSTMLayer, self).__init__()
        self.hidden_value_dim = hidden_value_dim
        self.hidden_type_dim = hidden_type_dim
        self.comp_linear_v = [nn.Linear(in_features=2 * hidden_value_dim,
                                       out_features=5 * hidden_value_dim) for i in range(0, 5)]
        self.comp_moe = nn.Linear(in_features=2 * hidden_type_dim,
                                  out_features=5)
        self.type_predictor = type_predictor
        self.binary_type_predictor = binary_type_predictor
        self.max_seq_len = max_seq_len
        self.decoder = decoder
        self.vocab_size = len(self.decoder.vocab)
        self.encoder = nn.GRU(self.vocab_size, self.hidden_value_dim, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(0, 5):
            init.orthogonal_(self.comp_linear_v[i].weight.data)
            init.constant_(self.comp_linear_v[i].bias.data, val=0)
        init.orthogonal_(self.comp_moe.weight.data)
        init.constant_(self.comp_moe.bias.data, val=0)
        self.encoder.reset_parameters()

        # type predictor's parameters are not reset

    def apply_decoder_template(self, dt, input_cat):
        # find the locations of the slots in the template:
        B, L, M = dt.size()
        s0 = self.decoder.vocab._token_to_idx("_0")
        dt_flat = dt.flatten()
        dt_slot_idx = torch.nonzero(dt_flat >= s0)F
        # TODO: assumes tokens have consecutive values starting from s0
        slot_idx = dt_flat[dt_slot_idx] - s0
        input_cat_flat = torch.flatten(input_cat, start_dim=0, end_dim=1)
        slot_values = input_cat_flat[:, slot_idx]
        dt_flat[dt_slot_idx] = slot_values

        return dt_flat.view(B, L, M)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l, d) tuple, where h_l and c_l have size
                (batch_size, max_length, hidden_value_dim + hidden_type_dim)
                and d has size (batch_size, max_seq_len)
            r: A (h_r, c_r, d) tuple, where h_r and c_r have size
                (batch_size, max_length, hidden_value_dim + hidden_type_dim)
                and d has size (batch_size, max_seq_len)
        Returns:
            h, c, d, hom_loss: [0, 1] The hidden and cell state of the composed parent,
                each of which has the size (batch_size, max_length - 1, hidden_value_dim + hidden_type_dim);
               [3] the decoding of that parent into the output vocabulary y_vocab (batch_size, max_seq_len, y_vocab size);
                and [4] the homomorphic loss.  The homomorphic loss is the sum of the losses for the abstraction
               homomorphism and the decoding homomorphism.
        """

        # extract state from left and right args
        hl, cl, dl = l
        hr, cr, dr = r

        # extract value and type information
        hl_v, hl_t = torch.split(hl, [self.hidden_value_dim, self.hidden_type_dim], dim=2)  # hl
        hr_v, hr_t = torch.split(hr, [self.hidden_value_dim, self.hidden_type_dim], dim=2)  # hr

        # TK DEBUG
        #print("type = ", hl_t.argmax(dim=-1))

        # compute updated hidden state and memory values
        hl_cat_t = torch.cat([hl_t, hr_t], dim=2)
        hlr_cat_v = torch.cat([hl_v, hr_v], dim=2)
        B, L, T = hl_v.size()
        h_v = torch.zeros(B, L, T, 5)
        for l in range(0, 5):
            treelstm_vector = self.comp_linear_v[l](hlr_cat_v)
            i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
            c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
                   + u.tanh() * i.sigmoid())
            h_v[:, :, :, l] = o.sigmoid() * c.tanh()

        moe_gates = torch.sigmoid(self.comp_moe(hl_cat_t))
        h_gated_v = torch.zeros_like(hl_v)
        for l in range(0, 5):
            h_gated_v = h_gated_v + h_v[:, :, :, l] * moe_gates[:, :, l].unsqueeze(2).repeat(1, 1, T)

        # DEBUG
        # print("gates: ", moe_gates[0, :, :])

        # compute updated hidden state and memory types
        hlr_cat_t = torch.cat([hl_t, hr_t], dim=2)
        _, h_t = self.binary_type_predictor(hlr_cat_t)

        # DEBUG
        # print("left type: ", hl_t[0, :, :])
        # print("right type: ", hr_t[0, :, :])

        # compute type prediction from semantic value
        _, sem_h_t = self.type_predictor(h_gated_v)

        # compute abstract homomorphic loss
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        abstraction_hom_loss = kl_loss(torch.clamp(h_t, min=1e-10, max=1.0).log(), sem_h_t)

        # compute output type
        new_h_type = F.one_hot(torch.argmax((h_t + sem_h_t) / 2.0, dim=-1), num_classes=self.hidden_type_dim)

        # compute output hidden state
        new_h = torch.cat([h_gated_v, new_h_type], dim=2)

        # compute concatenated input decodings: (B x L x M), (B x L x M) -> B x L x 2M
        # B = batch size, L = sentence length, M = max decoded seq len
        # last dimension holds logits
        d_cat = torch.cat([dl, dr], dim=2)

        # sample from them (argmax over last dimension)
        d_cat_sample = torch.argmax(d_cat, dim=-1)

        # represent in 1-hot: (BxL) x 2*max_seq_len x vocab size
        vocab_size = len(self.decoder.vocab)
        d_cat_sample = F.one_hot(d_cat_sample, num_classes=vocab_size)

        # flatten the batch and length dimensions for the encoder (set to batch first)
        # -> (B*L) x 2M x V
        d_cat_sample = torch.flatten(d_cat_sample, start_dim=0, end_dim=1).float()
        
        # encode the concatenated input decodings -> B x L x H (hidden)
        d_enc = self.encoder(d_cat_sample)[1].squeeze(0).view(B, L, self.hidden_value_dim)

        # decode the decoding template (a function specification):
        # of the form s1, s2, ..., sk where each si is either a target
        # vocabulary element or the index of an element of the (concatenated)
        # input pair.  This template is interpreted to produce the output
        # (either a target element or the indexed element from the input).
        # like a pointer network.
        # should be much easier for it to learn to do the syntactic operations
        # (on sequences) like `opposite`, `twice`, `thrice` etc.

        # flatten the first two dimensions for the decoder -> B*L x H
        d_enc = torch.flatten(d_enc, start_dim=0, end_dim=1)

        # decode -> B x L x M x V, where M is max seq len and V is the size of the (augmented) x vocab
        # last dimension holds logits
        dt = self.decoder.decode(d_enc, self.max_seq_len)[1].view(B, L, self.max_seq_len, self.vocab_size)

        # sample from it -> B x L x M
        dt_sample = torch.argmax(dt, dim=-1)

        # apply decoder template to create the decoding (substituting input values for slots)
        new_d = self.apply_decoder_template(dt_sample, d_cat_sample)

        # homomorphic alignment:
        #
        # decode composite
        dec_comp = self.decoder.decode(torch.flatten(h_gated_v, start_dim=0, end_dim=1),
                                       self.max_seq_len)[1].view(B, L, self.max_seq_len, self.vocab_size)

        # compute homomorphic loss for decoding:
        decoding_hom_loss = kl_loss(torch.log_softmax(new_d, dim=-1), torch.softmax(dec_comp, dim=-1))
        decoding = F.one_hot(torch.argmax((dec_comp + new_d) / 2.0, dim=-1),
                             num_classes=new_d.size(-1))

        # TK DEBUG
        print("new_d: ", dec_comp)

        hom_loss = abstraction_hom_loss + decoding_hom_loss

        return (new_h, c, decoding), hom_loss

class TypedBinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_value_dim, hidden_type_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature, bidirectional, max_seq_len, decoder, is_lstm=False, scan_token_to_type_map=None,
                 input_tokens=None):
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
        self.max_seq_len = max_seq_len
        self.decoder = decoder

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
            self.treelstm_layer = TypedBinaryTreeLSTMLayer(2 * hidden_value_dim, 2 * hidden_type_dim,
                                                           self.type_predictor,
                                                           self.binary_type_predictor,
                                                           self.max_seq_len,
                                                           self.decoder)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * (hidden_value_dim + hidden_type_dim)))
        else:
            self.treelstm_layer = TypedBinaryTreeLSTMLayer(hidden_value_dim, hidden_type_dim,
                                                           self.type_predictor,
                                                           self.binary_type_predictor,
                                                           self.max_seq_len,
                                                           self.decoder)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_value_dim + hidden_type_dim))

        self.reset_parameters()

    def reduce_gumbel_temp(self, it):
        self.type_predictor.reduce_gumbel_temp(0.0003, it, verbose=True)
        self.binary_type_predictor.reduce_gumbel_temp(0.0003, it, verbose=True)

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

        self.binary_type_predictor.reset_parameters()

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c, old_d = old_state
        new_h, new_c, new_d = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        done_mask = done_mask.unsqueeze(3)
        d = done_mask * new_d + (1 - done_mask) * old_d[:, :-1, :, :]
        return h, c, d

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c, new_d = new_state
        old_h, old_c, old_d = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        old_d_left, old_d_right = old_d[:, :-1, :, :], old_d[:, 1:, :, :]
        comp_weights = (self.comp_query * new_h).sum(-1)
        comp_weights = comp_weights / math.sqrt(self.hidden_value_dim + self.hidden_type_dim)
        if not self.is_lstm:
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

        # mask hidden state, memory, and decodings
        new_h, selected_h = self.mask(old_h_left, old_h_right, select_mask, new_h)
        new_c, _ = self.mask(old_c_left, old_c_right, select_mask, new_c)
        B, L, M, V = new_d.size()
        new_d, selected_d = self.mask(old_d_left.view(B, L, M*V),
                                      old_d_right.view(B, L, M*V),
                                      select_mask,
                                      new_d.view(B, L, M*V))
        return new_h, new_c, new_d.view(B, L, M, V), select_mask, selected_h

    @staticmethod
    def mask(old_left, old_right, select_mask, new):
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_right)
        new = (select_mask_expand * new
                 + left_mask_expand * old_left
                 + right_mask_expand * old_right)
        selected = (select_mask_expand * new).sum(1)
        return new, selected

    def forward(self, input, length, input_tokens, return_select_masks=False):
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
            B, L, T = h_v.size()

            # decode each word separately ((B*L), max_seq_len, target vocab size)
            # reshape to B x L x max_seq_len x len(decoder.vocab)
            target_vocab_size = len(self.decoder.vocab)     # includes max seq len slots variables "_i"
            B, L = input_tokens.size()
            initial_decodings = torch.zeros(B, L, self.max_seq_len, dtype=torch.int64)
            initial_decodings = F.one_hot(initial_decodings.view(B*L*self.max_seq_len),
                                          num_classes=target_vocab_size).view(B, L,
                                                                              self.max_seq_len,
                                                                              target_vocab_size)
            # self.decoder.decode(h_v.view(B*L, T), self.max_seq_len)
            # TK DEBUG
            # total_size = 1
            # for s in initial_decodings.shape:
            #     total_size = total_size*s
            # if total_size != B*L*self.max_seq_len*target_vocab_size:
            #     print("what?")
            # initial_decodings = initial_decodings.view(B, L, self.max_seq_len, target_vocab_size)

            if self.scan_token_to_type_map is not None:
                # compute cross entropy loss of predicted type against the oracle
                T_t = h_t.shape[2]
                target_types = self.scan_token_to_type_map[input_tokens.view(B*L)]
                hom_loss_sum = F.nll_loss(torch.clamp(h_t.view(B*L, T_t), min=1e-10, max=1.0).log(), target_types)

            new_types = F.one_hot(h_t_samples, num_classes=self.hidden_type_dim)
            h = torch.concat((h_v, new_types), dim=2)
            state = h, c, initial_decodings
        nodes = []
        if self.intra_attention:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c, d = state
            l = (h[:, :-1, :], c[:, :-1, :], d[:, :-1, :, :])
            r = (h[:, 1:, :], c[:, 1:, :], d[:, 1:, :, :])
            new_state, hom_loss = self.treelstm_layer(l=l, r=r)
            hom_loss_sum += hom_loss
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, new_d, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:])
                new_state = (new_h, new_c, new_d)
                select_masks.append(select_mask)
                if self.intra_attention:
                    nodes.append(selected_h)
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                nodes.append(state[0])
        h, c, d = state
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
            return h.squeeze(1), c.squeeze(1), d.squeeze(1), hom_loss_sum
        else:
            return h.squeeze(1), c.squeeze(1), d.squeeze(1), hom_loss_sum, select_masks
