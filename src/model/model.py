import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from src.model.treelstm import TypedBinaryTreeLSTM, FeedForward
from src.model.scan_data import scan_word_to_type, scan_token_to_type

from torch.profiler import profile, record_function, ProfilerActivity


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=10, device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        if hidden is None:
            torch.zeros(1, 1, self.hidden_size, device=self.device)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        

class Decoder(nn.Module):
    def __init__(self, vocab, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.vocab = vocab
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab.size())

    def forward(self, x, hidden):
        B = x.size()[0]
        D = self.input_dim
        O = self.vocab.size()
        x = self.embedding(x)
        out, hidden = self.gru(x.view(B, 1, D), hidden)
        out = self.fc(out)
        return out.view(B, O), hidden

    def decode(self, sem_f, max_seq_len, temp, force=None):
        B, _ = sem_f.size()
        y_vocab = self.vocab
        V = y_vocab.size()
        M = max_seq_len

        # first token is '<SOS>', first hidden is `sem_f` repeated in each RNN hidden layer
        inputs = torch.tensor([y_vocab.token_to_idx('<SOS>') for _ in range(B)], dtype=torch.long)
        hidden = sem_f.unsqueeze(0).repeat_interleave(self.num_layers, dim=0)

        unf_idxs = torch.arange(B)  # unfinished batch indices
        decoded = torch.full((B, M), 0)  # -1 represents an empty slot, used to later compute mask
        logits = torch.zeros((B, M, V))
        for t in range(M):

            outputs, hidden = self.forward(inputs, hidden)

            #decoded[unf_idxs, t] = decoded_idxs = torch.argmax(outputs, dim=1)
            # TK -- changed from argmax
            # probs = torch.exp(outputs)
            # decoded_idxs = torch.multinomial(probs, 1).squeeze(1)
            # decoded[unf_idxs, t] = decoded_idxs
            logit_outputs = torch.log_softmax(outputs, -1)
            decoded_idxs = torch.nn.functional.gumbel_softmax(logit_outputs,
                                                               tau=temp,
                                                               hard=True).argmax(-1)
            # decoded[unf_idxs, t] = decoded_idxs = torch.argmax(outputs, dim=1)
            decoded[unf_idxs, t] = decoded_idxs

            logits[unf_idxs, t, :] = logit_outputs  # save logits for loss computation
            logits[unf_idxs, t, :] = outputs  # save logits for loss computation

            if force is not None:
                if t + 1 < force.shape[1]:
                    inputs = force[:, t]
                else:
                    break  # break if we've reached the end of the forced input
            else:
                is_finished = decoded_idxs == y_vocab.token_to_idx('<EOS>')
                unf_idxs = unf_idxs[~is_finished]
                if len(unf_idxs) > 0:
                    inputs = decoded_idxs[~is_finished]
                    hidden = hidden[:, ~is_finished]
                else:
                    break  # break if all sequences have reached '<EOS>'

        # TK DEBUG
        # return decoded[:, :t + 1], logits[:, :t + 1, :]
        return decoded, logits


# class TemplateDecoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, max_length):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.max_length = max_length
#         self.fc = nn.Linear(hidden_dim, 2)
#         self.is_done = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         for i in range(self.max_length):
#
#         B = x.size()[0]
#         D = self.input_dim
#         O = 3
#         x = self.embedding(x)
#         out, hidden = self.gru(x.view(B, 1, D), hidden)
#         out = self.fc(out)
#         return out.view(B, O), hidden
#
#     def decode(self, sem_f, max_seq_len, temp, force=None):
#         B, _ = sem_f.size()
#         V = 3
#         M = max_seq_len
#
#         inputs = torch.tensor([2 for _ in range(B)], dtype=torch.long)
#         hidden = sem_f.unsqueeze(0).repeat_interleave(self.num_layers, dim=0)
#
#         unf_idxs = torch.arange(B)  # unfinished batch indices
#         decoded = torch.full((B, M), -1)  # -1 represents an empty slot, used to later compute mask
#         logits = torch.zeros((B, M, V))
#         for t in range(M):
#
#             outputs, hidden = self.forward(inputs, hidden)
#
#             #decoded[unf_idxs, t] = decoded_idxs = torch.argmax(outputs, dim=1)
#             # TK -- changed from argmax
#             # probs = torch.exp(outputs)
#             # decoded_idxs = torch.multinomial(probs, 1).squeeze(1)
#             # decoded[unf_idxs, t] = decoded_idxs
#             logit_outputs = torch.log_softmax(outputs, -1)
#             decoded_idxs = torch.nn.functional.gumbel_softmax(logit_outputs,
#                                                                       tau=temp,
#                                                                       hard=True).argmax(-1)
#             decoded[unf_idxs, t] = decoded_idxs
#
#             logits[unf_idxs, t, :] = logit_outputs  # save logits for loss computation
#
#             if force is not None:
#                 if t + 1 < force.shape[1]:
#                     inputs = force[:, t]
#                 else:
#                     break  # break if we've reached the end of the forced input
#             else:
#                 is_finished = decoded_idxs == 2
#                 unf_idxs = unf_idxs[~is_finished]
#                 if len(unf_idxs) > 0:
#                     inputs = decoded_idxs[~is_finished]
#                     # TK FIXME: what do we do with hidden?
#                     hidden = hidden[:, ~is_finished]
#                 else:
#                     break  # break if all sequences have reached '<EOS>'
#
#         # TK DEBUG
#         # return decoded[:, :t + 1], logits[:, :t + 1, :]
#         return decoded, logits


class CompositionalLearner(nn.Module):

    def __init__(self, model, y_vocab, x_vocab, word_dim, hidden_value_dim, hidden_type_dim,
                 decoder_hidden_dim, decoder_num_layers, use_leaf_rnn, bidirectional,
                 intra_attention, use_batchnorm, dropout_prob, max_y_seq_len, use_prim_type_oracle,
                 syntactic_supervision, dataset):
        super(CompositionalLearner, self).__init__()
        self.model = model
        self.num_classes = len(y_vocab)
        self.num_words = len(x_vocab)
        self.y_vocab = y_vocab
        self.x_vocab = x_vocab
        self.word_dim = word_dim
        self.hidden_value_dim = hidden_value_dim
        self.hidden_type_dim = hidden_type_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_layers = decoder_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.bidirectional = bidirectional
        self.intra_attention = intra_attention
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.max_y_seq_len = max_y_seq_len
        self.use_prim_type_oracle = use_prim_type_oracle
        self.syntactic_supervision = syntactic_supervision
        self.dropout = nn.Dropout(dropout_prob)
        self.embedding = nn.Embedding(num_embeddings=self.num_words,
                                      embedding_dim=word_dim)
        self.embedding.weight.requires_grad=False
        self.scan_token_to_type_map = None
        if self.syntactic_supervision:
            self.scan_token_to_type_map = torch.zeros([len(self.x_vocab)], dtype=torch.int64)
            for word in scan_token_to_type:
                if word not in self.x_vocab._token_to_idx:
                    continue
                token_idx = self.x_vocab._token_to_idx[word]
                type = scan_token_to_type[word]
                self.scan_token_to_type_map[token_idx] = type
        elif self.use_prim_type_oracle:
            self.scan_token_to_type_map = torch.zeros([len(self.x_vocab)], dtype=torch.int64)
            for word in scan_word_to_type:
                token_idx = self.x_vocab._token_to_idx[word]
                type = scan_word_to_type[word]
                self.scan_token_to_type_map[token_idx] = type

        self.decoder_sem = nn.ModuleList()
        # for 2 span
        self.decoder_sem.append(nn.ModuleList([FeedForward(decoder_hidden_dim, [decoder_hidden_dim], 8 * 3)
                                                    for _ in range(hidden_type_dim - 9)]))
        # for 3 span
        self.decoder_sem.append(nn.ModuleList([FeedForward(decoder_hidden_dim, [decoder_hidden_dim], 8 * 4)
                                                    for _ in range(hidden_type_dim - 9)]))
        # initial decodings
        self.decoder_init = Decoder(y_vocab, decoder_hidden_dim, decoder_hidden_dim, decoder_num_layers)
        # substitution templates
        self.decoder_sub = nn.ModuleList([FeedForward(decoder_hidden_dim, [decoder_hidden_dim], 6 * 21)
                                                    for _ in range(hidden_type_dim)])
        # model
        self.encoder = TypedBinaryTreeLSTM(word_dim=word_dim,
                                           hidden_value_dim=hidden_value_dim,
                                           hidden_type_dim=hidden_type_dim,
                                           use_leaf_rnn=use_leaf_rnn,
                                           intra_attention=intra_attention,
                                           gumbel_temperature=1.0,
                                           bidirectional=bidirectional,
                                           max_seq_len=self.max_y_seq_len,
                                           decoder_sem=self.decoder_sem,
                                           decoder_init=self.decoder_init,
                                           decoder_sub=self.decoder_sub,
                                           x_vocab=self.x_vocab,
                                           y_vocab=self.y_vocab,
                                           dataset=dataset,
                                           is_lstm=model == 'lstm',
                                           scan_token_to_type_map=self.scan_token_to_type_map)
        self.lstm_encoder = nn.LSTM(len(self.x_vocab), self.hidden_value_dim, batch_first=True)
        self.dataset = dataset
        self.reset_parameters()
        self.lstm_encoder.reset_parameters()

    def reduce_gumbel_temp(self, iter):
        self.encoder.reduce_gumbel_temp(iter)

    def reset_gumbel_temp(self, new_temp):
        self.encoder.reset_gumbel_temp(new_temp)

    def reset_parameters(self):
        init.normal_(self.embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()

    def forward(self, x, length, force=None, positions_force=None, types_force=None, spans_force=None):
        # TK DEBUG
        self.embedding.weight.requires_grad=False
        x_embed = self.embedding(x)

        # TK DEBUG
        # x_embed = self.dropout(x_embed)
        #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #with record_function("model_inference"):
        decoding= self.encoder(input=x_embed, length=length, input_tokens=x,
                                positions_force=positions_force,
                                types_force=types_force,
                                spans_force=spans_force)
        # DEBUG
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return decoding
