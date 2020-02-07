import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader_batch import FrEnParallelCorpusDataset, collate_fn_cuda
from networks.EncoderGRUMulti import EncoderGRU
from networks.DecoderDotAttnMulti import DecoderDotAttn
from metrics import evaluate_gru_multi
from model_utils import save_losses, save_model, load_model_dict
import progressbar


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def train_batch(batch_data, encoder_model, decoder_model, encoder_opt, decoder_opt, criterion):
    loss = 0
    source_tensor, source_lens, target_tensor, target_lens = batch_data
    encoder_output, hidden = encoder_model(source_tensor, source_lens)

    b_size = target_tensor.size(1)

    decoder_input = torch.tensor([1]*b_size, dtype=torch.int64,
                                 device=target_tensor.device).unsqueeze(0)
    decoder_hidden = hidden[:encoder_model.n_layers]

    max_len = max(target_lens)

    for t in range(max_len):
        decoder_output, decoder_hidden, decoder_attn = decoder_model(
            decoder_input, decoder_hidden, encoder_output
        )
        # output[t] = decoder_output
        decoder_input = target_tensor[t].unsqueeze(0)
        loss += criterion(decoder_output.squeeze(), target_tensor[t])

    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), 25)
    dc = torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), 25)

    encoder_opt.step()
    decoder_opt.step()

    return loss


def checkpoint_model(encoder, decoder, emb_dim, learning_rate, iterations):
    save_model(encoder, "n_encoder_gru_enc_multi_1000", emb_dim, learning_rate, iterations)
    save_model(decoder, "n_decoder_gru_enc_multi_1000", emb_dim*2, learning_rate, iterations)


if __name__ == "__main__":
    batch_size = 32
    word_embedding_dim = 1000
    use_cuda = torch.cuda.is_available()
    learning_rate = 1e-4
    epochs = 30
    loss_total = 0

    pc = FrEnParallelCorpusDataset("./data/train_f_bpe", "./data/train_e_bpe",
                                   ["./data/val_f_bpe", "./data/test_f_bpe"],
                                   ["./data/val_e_bpe", "./data/test_e_bpe"],
                                   use_cuda=use_cuda)

    validation_pc = FrEnParallelCorpusDataset("./data/val_f_bpe", "./data/val_e_bpe",
                                              use_cached_training_tuples=False,
                                              use_cuda=use_cuda)

    validation_dataloader = DataLoader(dataset=validation_pc,
                                       batch_size=1,
                                       collate_fn=collate_fn_cuda(use_cuda),
                                       shuffle=True)

    training_dataloader = DataLoader(dataset=pc,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn_cuda(use_cuda),
                                     shuffle=True)

    encoder = EncoderGRU(input_size=pc.fr_vocab_size,
                         hidden_size=word_embedding_dim)

    decoder = DecoderDotAttn(output_size=pc.en_vocab_size,
                             hidden_size=word_embedding_dim,
                             use_cuda=use_cuda)

    criterion = torch.nn.NLLLoss(ignore_index=1)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    total_samples = pc.__len__()
    pbar_size = total_samples + (batch_size - (total_samples % batch_size))

    loss_per_iter = []
    bleu_per_iter = []
    best_bleu = 0

    # Initialize the progress bar
    widgets = ["Training Positional Encoder/Decoder:", progressbar.Percentage(),
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=pbar_size, widgets=widgets).start()

    for iteration in range(epochs):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        processed = 0

        for batch in training_dataloader:
            batch_loss = train_batch(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            loss_total += batch_loss.item()
            processed += batch_size
            pbar.update(processed)

        print(f"Loss iteration {iteration+1}: {loss_total}")
        bleu_score = evaluate_gru_multi(encoder, decoder, validation_dataloader, validation_pc)
        print(f"Bleu Score {iteration+1}: {bleu_score}")
        bleu_per_iter.append(bleu_score)

        if bleu_score > best_bleu:
            checkpoint_model(encoder, decoder, word_embedding_dim, learning_rate, iteration)
            best_bleu = bleu_score

        loss_per_iter.append(loss_total)
        loss_total = 0

    save_losses(f"models/n_loss_training_gru_enc_multi_1000.txt", loss_per_iter)
    save_losses(f"models/n_bleu_validation_gru_enc_multi_1000.txt", bleu_per_iter)
    save_model(encoder, "n_encoder_gru_enc_multi_1000", word_embedding_dim, learning_rate, epochs)
    save_model(decoder, "n_decoder_gru_enc_multi_1000", word_embedding_dim*2, learning_rate, epochs)
