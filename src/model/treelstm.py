import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import init, Embedding
import torch.nn.functional as F
from torch.nn import functional as F

from . import basic
import numpy as np
from src.scan.data import get_decoding_force

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
    def __init__(self, hidden_value_dim, hidden_type_dim, type_predictor, binary_type_predictor,
                 max_seq_len, decoder_sem, decoder_init, gumbel_temperature):
        super(TypedBinaryTreeLSTMLayer, self).__init__()
        self.hidden_value_dim = hidden_value_dim
        self.hidden_type_dim = hidden_type_dim
        # No need of functions for input types
        self.comp_linear_v = nn.ModuleList([nn.Linear(in_features=2 * (hidden_value_dim + hidden_type_dim),
                                                      out_features=5 * hidden_value_dim) for i in range(hidden_type_dim - 9)])
        self.type_predictor = type_predictor
        self.binary_type_predictor = binary_type_predictor
        self.max_seq_len = max_seq_len
        #self.decoder_sem = decoder_sem
        self.decoder_sem = nn.ModuleList([nn.Linear(in_features=hidden_value_dim,
                                                    out_features=24) for i in range(hidden_type_dim - 9)])
        self.decoder_init = decoder_init
        self.vocab_size = len(self.decoder_init.vocab)
        self.encoder = nn.GRU(self.vocab_size, self.hidden_value_dim, batch_first=True)
        self.gumbel_temperature = gumbel_temperature
        self.type_embedding = nn.Embedding(num_embeddings=self.hidden_type_dim,
                                           embedding_dim=self.hidden_value_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.hidden_type_dim - 9):
            init.orthogonal_(self.comp_linear_v[i].weight.data)
            #init.orthogonal_(self.comp_linear_t.weight.data)    # TODO: is this what we want?
            init.constant_(self.comp_linear_v[i].bias.data, val=0)
            #init.constant_(self.comp_linear_t.bias.data, val=0)
            # type predictor's parameter are not reset

    def apply_decoder_template(self, dt_sample, input_cat):
        # dt - B x L x K x 2
        # dt_sample - B x L x K
        # input_cat - B x L x 2M x V
        # K is max template sequence length which will be smaller
        # than M the max decoding length

        # sample concatenated input value pair -> B x L x 2M
        # input_cat_sample = input_cat.argmax(-1)

        # extract the arguments from the input
        il = input_cat[:, :, :self.max_seq_len, :]
        ir = input_cat[:, :, self.max_seq_len:, :]
        B, L, _ = dt_sample.size()
        input_cat_sample = input_cat.argmax(-1)
        dl_sample = input_cat_sample[:, :, :self.max_seq_len]
        dr_sample = input_cat_sample[:, :, self.max_seq_len:]

        # compose the result by selecting either left or right input data based
        # on the template
        K = dt_sample.size(2)
        M = dl_sample.size(2)
        V = input_cat.size(3)
        result = torch.zeros(B, L, M, V)
        for i in range(B):
            for k in range(L):
                length_l = torch.count_nonzero(dl_sample[i][k]).tolist()
                length_r = torch.count_nonzero(dr_sample[i][k]).tolist()
                idx = 0
                for t in range(K):  # template index
                    template_code = dt_sample[i][k][t].tolist()
                    if length_l != 0 and idx+length_l <= M and template_code == 0:     # left
                        result[i][k][idx:idx+length_l][:] = il[i][k][:length_l][:]
                        idx = idx+length_l
                    elif length_r != 0 and idx+length_r <= M and template_code == 1:   # right
                        result[i][k][idx:idx+length_r][:] = ir[i][k][:length_r][:]
                        idx = idx+length_r
                    elif idx < M and template_code == 2:
                        result[i][k][idx:][0] = 1.0
                        break
        return result


    def forward(self, l=None, r=None, positions_force=None, target_types=None):
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
               [3] the decoding of that parent into the output vocabulary (batch_size, max_seq_len, vocab size);
                and [4] the homomorphic loss.  The homomorphic loss is the sum of the losses for the abstraction
               homomorphism and the decoding homomorphism.
        """

        # extract state from left and right args
        hl, cl, dl = l
        hr, cr, dr = r

        # extract value and type information
        hl_v, hl_t = torch.split(hl, [self.hidden_value_dim, self.hidden_type_dim], dim=2)  # hl
        hr_v, hr_t = torch.split(hr, [self.hidden_value_dim, self.hidden_type_dim], dim=2)  # hr

        # compute updated hidden state and memory values
        # TK changed to include type information
        hlr_cat_v = torch.cat([hl, hr], dim=2)
        if target_types is not None:
            B, L, _ = hlr_cat_v.size()
            treelstm_vector = torch.zeros(B, L, 5 * self.hidden_value_dim)
            for i in range(B):
                treelstm_vector[i] = self.comp_linear_v[target_types[i].int() - 9](hlr_cat_v[i])
        else:
            treelstm_vector = self.comp_linear_v[0](hlr_cat_v)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
               + u.tanh() * i.sigmoid())
        h_v = o.sigmoid() * c.tanh()

        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        ce_loss = torch.nn.CrossEntropyLoss()
        if target_types != None:
            _, new_h_type = self.type_predictor(h_v)
            B, L, T = new_h_type.shape
            positions = torch.zeros(B).long()
            for i in range(B):
                if positions_force[i] == []:
                    positions[i] = 0 # default to start
                else:
                    positions[i] = positions_force[i][-1]
            abstraction_hom_loss = F.cross_entropy(torch.clamp(new_h_type[torch.arange(B), positions].view(B, T),
                                                               min=1e-10, max=1.0).log(),
                                       target_types.long())
            new_h = torch.cat([h_v, new_h_type], dim=2)
        else:
            hlr_cat_t = torch.cat([hl_t, hr_t], dim=2)
            _, h_t = self.binary_type_predictor(hlr_cat_t)

            # DEBUG
            # print("left type: ", hl_t[0, :, :])
            # print("right type: ", hr_t[0, :, :])

            # predict output type from semantic value
            _, sem_h_t = self.type_predictor(h_v)

            # compute abstract homomorphic loss to reduce their difference
            abstraction_hom_loss = kl_loss(h_t.log(), sem_h_t)

            # take output type to be the average of the two predictions
            new_h_type = (h_t + sem_h_t) / 2.0

            # compute output hidden state
            new_h = torch.cat([h_v, new_h_type], dim=2)

        new_h_type_idx = new_h_type.argmax(-1).long()
        B, L, T = new_h_type.shape
        #new_h_type_idx = F.gumbel_softmax(type_logits, tau=self.gumbel_temperature, dim=-1, hard=True).argmax(-1).long()
        type_embedding = self.type_embedding(target_types.unsqueeze(1).repeat(1, L))
        # TK FIXME -- put these in the args somewhere K=8, template_vocab_size=3
        K = 8
        dt = torch.zeros(B, L, K, 3)
        # hard code start template
        start_template = torch.full((L, K), 2)
        start_template[:, 0] = 0
        start_template = F.one_hot(start_template, 3)
        for i in range(B):
            #_, dt[i] = self.decoder_sem[target_types[i]-9].decode(type_embedding[i], K)
            dt[i] = self.decoder_sem[target_types[i]-9](type_embedding[i]).view(L, K, 3)
            # hard code start template
            if target_types[i] == 25:
                dt[i] = start_template
        dt_sample = dt.argmax(-1)

        # compute concatenated input decodings: (B x L x M x V), (B x L x M) -> B x L x 2M x V
        # B = batch size, L = sentence length, M = max decoded seq len
        # last dimension holds logits
        d_cat = torch.cat([dl, dr], dim=2).float()
        new_d = self.apply_decoder_template(dt_sample, d_cat)
        # decoding = F.one_hot(torch.flatten(new_d, start_dim=0, end_dim=1),
        #                      num_classes=self.vocab_size).view(B, L, self.max_seq_len, self.vocab_size)

        return (new_h, c, new_d), abstraction_hom_loss, dt

class TypedBinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_value_dim, hidden_type_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature, bidirectional, max_seq_len, decoder_sem, decoder_init, is_lstm=False,
                 scan_token_to_type_map=None, input_tokens=None, positions_force=None, types_force=None):
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
        self.decoder_sem = decoder_sem
        self.decoder_init = decoder_init
        self.positions_force = positions_force
        self.types_force = types_force
        self.initial_decoder = torch.nn.Linear(hidden_value_dim, len(self.decoder_init.vocab))

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
                                                           self.decoder_sem,
                                                           self.decoder_init,
                                                           self.gumbel_temperature)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * (hidden_value_dim + hidden_type_dim)))
        else:
            self.treelstm_layer = TypedBinaryTreeLSTMLayer(hidden_value_dim, hidden_type_dim,
                                                           self.type_predictor,
                                                           self.binary_type_predictor,
                                                           self.max_seq_len,
                                                           self.decoder_sem,
                                                           self.decoder_init,
                                                           self.gumbel_temperature)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_value_dim + hidden_type_dim))

        self.reset_parameters()

    def reduce_gumbel_temp(self, it):
        # TK DEBUG 0.0003 -> 0.00003
        self.type_predictor.reduce_gumbel_temp(0.001, it, verbose=True)
        self.binary_type_predictor.reduce_gumbel_temp(0.001, it, verbose=True)

    def reset_gumbel_temp(self, new_temp):
        self.type_predictor.gumbel_temp = new_temp
        self.binary_type_predictor.gumbel_temp = new_temp

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
        old_h, old_c, old_d = old_state
        new_h, new_c, new_d = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        done_mask = done_mask.unsqueeze(3)
        d = done_mask * new_d + (1 - done_mask) * old_d[:, :-1, :, :]
        return h, c, d

    def select_composition(self, old_state, new_state, mask, positions_force=None):
        new_h, new_c, new_d = new_state
        old_h, old_c, old_d = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        old_d_left, old_d_right = old_d[:, :-1, :, :], old_d[:, 1:, :, :]

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

    def forward(self, input, length, input_tokens, return_select_masks=False, positions_force=None, types_force=None):
        max_depth = input.size(1)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)
        select_masks = []
        hom_loss_sum = 0
        #print("input token:", input_tokens[0]) # debug
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
            # decode each word separately ((B*L), max_seq_len, target vocab size)
            # reshape to B x L x max_seq_len x len(decoder.vocab)
            target_vocab_size = len(self.decoder_init.vocab)
            B, L, M = input.size()
            if self.scan_token_to_type_map is not None:
                # compute cross entropy loss of predicted type against the oracle
                B, L, T_t = h_t.shape
                target_types = self.scan_token_to_type_map[input_tokens.view(B*L)]
                hom_loss_sum = F.cross_entropy(torch.clamp(h_t.view(B*L, T_t), min=1e-10, max=1.0).log(), target_types)
            # _, initial_decodings = self.decoder_init.decode(torch.flatten(input, start_dim=0, end_dim=1),
            #
            #                                                 self.max_seq_len)
            # TK DEBUG FIXME: get the 49 from somewhere
            initial_decodings = torch.zeros(B, L, 49, target_vocab_size)
            initial_decodings[:, :, 0, :] = self.initial_decoder(h_v).softmax(-1)
            pad_decoding = torch.tensor([0 for _ in range(target_vocab_size)])
            pad_decoding[0] = 1
            for i in range(B):
                for j in range(L):
                    if target_types[i*L+j] not in [0, 4]:
                        initial_decodings[i, j, 0, :] = pad_decoding
            # initial_decodings = initial_decodings.view(B, L, self.max_seq_len, target_vocab_size)
            # initial_decodings = F.one_hot(initial_decodings.view(B * L * self.max_seq_len).long(),
            #                               num_classes=target_vocab_size).view(B, L,
            #                                                                   self.max_seq_len,
            #                                                                   target_vocab_size)
            dt_all = None
            
            # TK DEBUG
            # new_types = F.one_hot(h_t_samples, num_classes=self.hidden_type_dim)
            # h = torch.concat((h_v, new_types), dim=2)
            h = torch.concat((h_v, h_t), dim=2)
            state = h, c, initial_decodings
        nodes = []
        if self.intra_attention:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c, d = state
            l = (h[:, :-1, :], c[:, :-1, :], d[:, :-1, :, :])
            r = (h[:, 1:, :], c[:, 1:, :], d[:, 1:, :, :])
            B, _, _ = h.size()
            if types_force != None:
                # get target types for this step and remove them
                target_types = torch.tensor([25 for i in range(B)]) # default to start
                for k in range(len(types_force)):
                    if types_force[k] != []:
                        target_types[k] = types_force[k][-1]
                        types_force[k].pop(-1)
            else:
                target_types = None

            new_state, hom_loss, dt = self.treelstm_layer(l=l, r=r, positions_force=positions_force,
                                                                                  target_types=target_types)
            hom_loss_sum += hom_loss
            if dt_all is None:
                dt_all = dt
            else:
                dt_all = torch.concat((dt_all, dt), dim=1)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, new_d, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:], positions_force=positions_force)
                # remove positions at this step
                if positions_force != None:
                    for k in range(len(positions_force)):
                        if positions_force[k] != []:
                            positions_force[k].pop(-1)
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
            return h.squeeze(1), c.squeeze(1), d.squeeze(1), hom_loss_sum, dt_all, initial_decodings
        else:
            return h.squeeze(1), c.squeeze(1), d.squeeze(1), hom_loss_sum, dt_all, initial_decodings, select_masks
