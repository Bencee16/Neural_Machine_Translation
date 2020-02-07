import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import progressbar

from networks.EncoderPositional import EncoderPositional
from networks.DecoderDotGRU import DecoderDotGRU
from data_loader import FrEnParallelCorpusDataset, SubsetSampler


def train_sentence(source_tensor, target_tensor,
                   encoder, decoder,
                   encoder_opt, decoder_opt,
                   loss_fn):
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    sent_loss = 0

    encoder_output = encoder(source_tensor)

    decoder_input = torch.tensor([[1]], device=encoder_output.device)

    decoder_hidden = torch.mean(encoder_output, dim=0).unsqueeze(0).unsqueeze(0)

    target_length = len(target_tensor)

    for ind in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_output
        )

        # create one hot
        target_ind = target_tensor[ind]
        target_vec = torch.tensor([target_ind], dtype=torch.int64, device=encoder_output.device)
        sent_loss += loss_fn(decoder_output, target_vec)
        decoder_input = torch.tensor([[target_ind]], device=encoder_output.device)

    sent_loss.backward()

    encoder_opt.step()
    decoder_opt.step()

    return sent_loss.item() / target_length


if __name__ == "__main__":
    sample_subset = True
    subset_size = 96
    num_epochs = 3
    batch_size = 32
    word_embedding_dim = 500
    learning_rate = 0.01
    use_cuda = torch.cuda.is_available()
    loss_total = 0

    pc = FrEnParallelCorpusDataset("./data/train_f_bpe", "./data/train_e_bpe",
                                   use_cuda=use_cuda)
    training_dataloader = DataLoader(dataset=pc,
                                     batch_size=batch_size,
                                     collate_fn=lambda x: x,
                                     shuffle=True)

    total_samples = 28000
    if sample_subset:
        subset_sampler = SubsetSampler(subset_size)
        total_samples = subset_size
        training_dataloader = DataLoader(dataset=pc,
                                         batch_size=batch_size,
                                         collate_fn=lambda x: x,
                                         sampler=subset_sampler,
                                         shuffle=False)

    encoder = EncoderPositional(vocab_size=pc.fr_vocab_size,
                                position_size=100,
                                word_emb_dim=word_embedding_dim,
                                positional_emb_dim=word_embedding_dim,
                                use_cuda=use_cuda)
    decoder = DecoderDotGRU(vocab_size=pc.en_vocab_size,
                            hidden_dim=word_embedding_dim*2,
                            use_cuda=use_cuda)
    criterion = nn.NLLLoss()
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Initialize the progress bar
    widgets = ["Training Positional Encoder/Decoder:", progressbar.Percentage(),
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=total_samples, widgets=widgets).start()

    for iter in range(num_epochs):
        processed = 0
        total = 28000

        for sample_batch in training_dataloader:
            for f_sent, e_sent in sample_batch:
                loss = train_sentence(f_sent, e_sent,
                                      encoder, decoder,
                                      encoder_optimizer, decoder_optimizer,
                                      criterion)
                loss_total += loss
                processed += 1
                pbar.update(processed)
        print(f"Loss iteration {iter+1}: {loss_total}")
        loss_total = 0

