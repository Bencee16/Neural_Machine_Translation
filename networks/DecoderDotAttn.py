import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextAttn(nn.Module):
    def __init__(self, hidden_size):
        super(ContextAttn, self).__init__()

        self.hidden_size = hidden_size

    def forward(self, input, encoder_values):
        # rotate to batch first
        input_bf = input.transpose(0, 1)
        enc_val_bf = encoder_values.transpose(0, 1)
        # print(f"input: {input_bf.size()}")
        # print(f"enc: {enc_val_bf.size()}")
        alignment = torch.bmm(enc_val_bf, input_bf.transpose(1, 2))
        alpha = F.softmax(alignment, dim=1)
        context_vec = torch.bmm(torch.transpose(alpha, 1, 2), enc_val_bf)
        # rotate batch to seq len first
        context_vec = context_vec.transpose(0, 1)
        # print(alignment.size())
        # print(alpha.size())
        return context_vec, alpha


class DecoderDotAttn(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob=0.1, use_cuda=False):
        super(DecoderDotAttn, self).__init__()

        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.attn_projection = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.attn = ContextAttn(hidden_size)

    def forward(self, input, prev_hidden_state, encoder_output_tensor):
        # print(input.size())
        # embeddings 1 x B x N
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # print(embedded.size())
        # print(len(input_lens))

        context, alpha = self.attn(prev_hidden_state, encoder_output_tensor)

        # print(embedded.size())
        total_hidden = torch.cat((embedded, context), dim=2)
        # print(total_hidden.size())
        total_hidden = F.relu(self.attn_projection(total_hidden))
        pre_output, hidden_state = self.gru(total_hidden, prev_hidden_state)
        # print(f"Target size: {emb_output.size()}")
        # total_hidden = torch.cat((emb_output, context), dim=2)
        # pre_output = F.relu(self.attn_projection(total_hidden))
        # print(pre_output.size())

        output = F.log_softmax(self.output_layer(pre_output), dim=2)
        # output = self.output_layer(pre_output)
        # print(output.size())
        return output, hidden_state, alpha
