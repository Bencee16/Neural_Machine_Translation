import torch
from torch import nn


class EncoderPositional(nn.Module):
    def __init__(self, vocab_size, position_size, word_emb_dim, positional_emb_dim, use_cuda=False):
        super(EncoderPositional, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim)
        self.positional_embeddings = nn.Embedding(position_size, positional_emb_dim)

        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()

    def forward(self, input):
        positions = torch.arange(len(input), dtype=torch.int64, device=input.device)
        words_embedded = self.word_embeddings(input)
        pos_embedded = self.positional_embeddings(positions)
        concatenated = torch.cat((words_embedded, pos_embedded), dim=1)
        return concatenated


class EncoderPositionalB(nn.Module):
    def __init__(self, vocab_size, position_size, word_emb_dim, positional_emb_dim, use_cuda=False):
        super(EncoderPositionalB, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, word_emb_dim)
        self.positional_embeddings = nn.Embedding(position_size, positional_emb_dim)

        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()

    def forward(self, input, input_lens):
        # positions max_len x 1
        positions = torch.arange(len(input), dtype=torch.int64, device=input.device)
        # positions max_len x B
        positions = torch.stack([positions for _ in range(input.size(1))], dim=1)
        # embeddings max_len x B x N
        words_embedded = self.word_embeddings(input)
        pos_embedded = self.positional_embeddings(positions)

        # concatenated max_len x B x 2N
        # concatenate word and positional embeddings
        concatenated = torch.cat((words_embedded, pos_embedded), dim=2)

        packed = torch.nn.utils.rnn.pack_padded_sequence(concatenated, input_lens)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        # also compute mean using packed data
        # mean B x 2N
        mean = torch.sum(output, dim=0)
        weights = torch.tensor([[l for _ in range(mean.size(1))] for l in input_lens], dtype=torch.float, device=mean.device)
        mean = torch.div(mean, weights)
        return output, mean
