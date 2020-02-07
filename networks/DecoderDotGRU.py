import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderDotGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim, dropout_prob=0.1, length_lim=50, use_cuda=False):
        super(DecoderDotGRU, self).__init__()

        self.use_cuda = use_cuda


        self.vocab_size = vocab_size
        self.emb_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        # Limit of generated tokens without a stop word
        self.length_limit = length_lim

        self.word_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_to_attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.attention_projection = nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)


    def forward(self, input, hidden_state, encoder_output_tensor):
        embedded_word = self.word_embedding(input).view(1, 1, -1)
        embedded_word = self.dropout(embedded_word)

        h_before_c = self.emb_to_attn(torch.cat((hidden_state[0], embedded_word[0]), dim=1))
        a = torch.mm(encoder_output_tensor, torch.transpose(h_before_c, 0, 1))
        alpha = F.softmax(a, dim=1)
        context_vec = torch.mm(torch.transpose(alpha, 0, 1), encoder_output_tensor)
        total_hidden = torch.cat((context_vec, h_before_c), dim=1)
        new_input = self.attention_projection(total_hidden)
        new_input = F.relu(new_input.unsqueeze(0))
        hidden_state = torch.transpose(hidden_state, 0, 1)

        output, hidden = self.gru(new_input, hidden_state)
        vocab_output = F.log_softmax(self.output_layer(output[0]), dim=1)
        return vocab_output, hidden

    def init_hidden(self):
        if self.use_cuda:
            return torch.zeros(1, 1, self.hidden_dim, device=torch.device("cuda"))
        return torch.zeros(1, 1, self.hidden_dim)
