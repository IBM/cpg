import math
import numpy as np

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from src.model.basic import int_to_one_hot


class CopyTemplate(nn.Module):

    def __init__(self, y_vocab, template_dim, template_len, max_span, type_dim):
        super(CopyTemplate, self).__init__()
        self.y_vocab = y_vocab
        self.template_dim = template_dim
        self.template_len = template_len
        self.max_span = max_span
        self.type_dim = type_dim

        self.decoder_copy = nn.ModuleList([nn.Linear(type_dim, template_len * (max_span+1)) for _ in range(type_dim)])
        self.templates = nn.ParameterDict()
        self.templates_current = nn.ParameterDict()

    def record_templates(self):
        self.templates.update(self.templates_current)
        self.templates_current = nn.ParameterDict()

    def generate_template(self, new_types, spans, gumbel_temp, concat=False):
        B = len(new_types)
        template_copy = torch.zeros(B, self.template_len, self.max_span+1)
        if concat: # simply concatenate input decodings
            for i in range(B):
                s = spans[i]
                for k in range(s):
                    template_copy[i, k] = int_to_one_hot(k+1, self.max_span+1)
        else:
            for i in range(B):
                s = spans[i]
                new_type = new_types[i].item()
                if str(new_type) in self.templates.keys():
                    template_copy[i, :, :s+1] = self.templates[str(new_type)]
                else:
                    template_logits = self.decoder_copy[new_type](torch.ones(self.type_dim))[:self.template_len * (s+1)].view(self.template_len, s+1)
                    template_copy[i, :, :s+1] = F.gumbel_softmax(template_logits.log_softmax(-1), tau=gumbel_temp, hard=True)
                    self.templates_current[str(new_type)] = template_copy[i, :, :s+1]
        return template_copy

    def apply_template(self, input_decodings, spans, template):
        B, N, M, V = input_decodings.size()
        result = torch.zeros(B, M, V)
        pad_vector = int_to_one_hot(self.y_vocab.token_to_idx('<PAD>'), M*V)
        for i in range(B):
            idx = 0
            # extract the arguments from the input
            for t in range(self.template_len):  # template index
                template_code = template[i, t, :spans[i]+1].unsqueeze(0).float()
                choices = [input_decodings[i][n].flatten() for n in range(spans[i])]
                choices.insert(0, pad_vector.flatten())
                choices = torch.stack(choices, dim=0)
                output = torch.mm(template_code, choices).view(M, V)
                # copy non-trailing pad part of the output
                output_idx = output.argmax(-1)
                output_len = 0
                for j in range(M):
                    if output_idx[j] == 0:
                        break
                    output_len = j+1
                output_len = min(output_len, M-idx)
                result[i][idx:idx+output_len] = output[:output_len]
                idx = idx + output_len
        return result


class SubstitutionTemplate(nn.Module):

    def __init__(self, y_vocab, template_dim, template_len, term_list_len, type_dim):
        super(SubstitutionTemplate, self).__init__()
        self.y_vocab = y_vocab
        self.template_dim = template_dim
        self.template_len = template_len
        self.term_list_len = term_list_len
        self.type_dim = type_dim

        self.decoder_sub = nn.ModuleList([nn.Linear(template_dim, template_len * (term_list_len+1)) for _ in range(type_dim)])
        self.templates = nn.ParameterDict()
        self.templates_current = nn.ParameterDict()

    def record_templates(self):
        self.templates.update(self.templates_current)
        self.templates_current = nn.ParameterDict()

    def generate_template(self, new_types, template_idx, gumbel_temp, predict_zero):
        B = len(new_types)
        temp_sub = torch.zeros(B, self.template_len, self.term_list_len+1)
        for i in range(B):
            new_type = new_types[i].item()
            idx = template_idx[i]
            # get recorded substitution templates
            if str(new_type) in self.templates.keys():
                temp_sub[i, :, :idx+1] = self.templates[str(new_type)][:, :idx+1]
            # get predicted substitution templates
            else:
                template = self.decoder_sub[new_type](torch.ones(self.template_dim)).view(self.template_len, self.term_list_len+1)
                if not predict_zero:
                    template[:, 0] = torch.ones(self.template_len).mul(-float("Inf"))
                temp_sub[i, :, :idx+1] = F.gumbel_softmax(template[:, :idx+1].log_softmax(-1), tau=gumbel_temp, hard=True)
                self.templates_current[str(new_type)] = temp_sub[i]
        return temp_sub

    def apply_template(self, decodings, terms, template):
        B, M, V = decodings.size()
        y_vector = int_to_one_hot(self.y_vocab.token_to_idx('y'), V)
        for i in range(B):
            # get choices
            choices = [y_vector.flatten()]
            for n in range(self.term_list_len):
                choices.append(terms[i, n])
            choices = torch.stack(choices, dim=0)
            # get term list ordered according to the substitution template
            K = self.template_len
            var_sub = torch.zeros(K, V)
            for k in range(K):
                var_sub[k] = torch.mm(template[i, k].unsqueeze(0), choices)
            # subsitute into slots
            idx = 0
            for j in range(M):
                if torch.equal(decodings[i, j], y_vector):
                    y_idx = decodings[i, j, self.y_vocab.token_to_idx('y')].repeat(V)
                    decodings[i, j] = torch.mul(var_sub[idx], y_idx)
                    # break if we reach the end of term list
                    if idx == K-1:
                        break
                    idx += 1
        return decodings


class CompositionalLearner(nn.Module):

    def __init__(self, dataset, gumbel_temp):
        super(CompositionalLearner, self).__init__()
        self.dataset = dataset
        self.gumbel_temp = gumbel_temp

    def reset_hyperparameters(self, train_accuracy_stage, iteration_stage):
        self.gumbel_temp = max(10.0 - train_accuracy_stage * 10, 0.5)
        self.dataset.reset_hyperparameters(iteration_stage)

    def start_eval(self):
        self.gumbel_temp = 1e-10
        self.dataset.reset_hyperparameters(0, eval=True)

    def record_templates(self):
        self.dataset.record_templates()

    def forward(self, input, positions, types, spans):
        # get initial decodings and terms
        decodings, terms = self.dataset.get_initial_dec_term(input, self.gumbel_temp)
        B, L, M, V = decodings.size()

        for t in range(L - 1):
            # get target types
            new_types, types = self.dataset.get_new_types(types)

            # get positions and spans
            new_positions, new_spans, positions, spans = self.dataset.get_pos_span(positions, spans)
            
            # compose input decodings and terms
            N = max(new_spans)
            input_decodings = torch.zeros(B, N, M, V)
            input_terms = torch.zeros(B, N, self.dataset.term_list_len, V)
            for i in range(B):
                s = new_spans[i].item()
                p = new_positions[i].item()
                input_decodings[i, :s] = decodings[i, p:p+s].view(s, M, V)
                input_terms[i, :s] = terms[i, p:p+s]
            output_decodings, output_terms = self.dataset.transform(input_decodings, input_terms, new_types, new_spans, self.gumbel_temp)

            # swap in the output decodings and terms
            _, L, _, _ = decodings.size()
            new_d = torch.zeros(B, L, M, V)
            new_t = torch.zeros(B, L, self.dataset.term_list_len, V)
            for i in range(B):
                s = new_spans[i]
                p = new_positions[i].unsqueeze(0)
                new_d[i, :p] = decodings[i, :p]
                new_d[i, p] = output_decodings[i]
                new_d[i, p+1:L-s+1] = decodings[i, p+s:L]
                new_t[i, :p] = terms[i, :p]
                new_t[i, p] = output_terms[i]
                new_t[i, p+1:L-s+1] = terms[i, p+s:L]
            decodings = new_d
            terms = new_t

        decodings = decodings[:, 0]
        decodings = self.dataset.normalize(decodings)

        return decodings