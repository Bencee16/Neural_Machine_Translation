import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader_batch import FrEnParallelCorpusDataset, collate_fn_cuda
from networks.EncoderPositional import EncoderPositionalB
from networks.EncoderGRU import EncoderGRU
from networks.DecoderDotAttn import DecoderDotAttn
from metrics import evaluate
from model_utils import save_losses, save_model, load_model_dict
import progressbar


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def train_batch(batch_data, encoder_model, decoder_model, encoder_opt, decoder_opt, criterion):
    loss = 0
    source_tensor, source_lens, target_tensor, target_lens = batch_data
    encoder_output, emb_means = encoder_model(source_tensor, source_lens)
    # encoder_output, hidden = encoder_model(source_tensor, source_lens)

    b_size = target_tensor.size(1)

    decoder_input = torch.tensor([1]*b_size, dtype=torch.int64,
                                 device=target_tensor.device).unsqueeze(0)
    # decoder_hidden = hidden[:2]
    decoder_hidden = emb_means.unsqueeze(0)

    max_len = max(target_lens)

    for t in range(max_len):
        decoder_output, decoder_hidden, decoder_attn = decoder_model(
            decoder_input, decoder_hidden, encoder_output
        )
        # output[t] = decoder_output
        decoder_input = target_tensor[t].unsqueeze(0)
        loss += criterion(decoder_output.squeeze(), target_tensor[t])

    # output, _, _ = decoder_model(sorted_target, sorted_target_lens, decoder_hidden, encoder_output)
    # print(output.size())

    # # Set through the properly sorted tensors to compute the loss
    # for ind in range(len(target_lens)):
    #     unsorted_ind = indx[ind]
    #     targets = target_tensor[:, unsorted_ind]
    #     t_len = target_lens[unsorted_ind]
    #
    #     # Slice valid sentence and remove SOS token from consideration
    #     sliced_targets = targets[1:t_len]
    #     # add EOS target
    #     sliced_targets = torch.cat((sliced_targets, torch.tensor([0], device=sliced_targets.device)))
    #
    #     predictions = output[0:t_len, ind, :]
    #     topv, topi = predictions.topk(1, dim=1)
    #     top_inds = topi.cpu().numpy().squeeze()
    #     target_inds = sliced_targets.cpu().numpy().squeeze()
    #     print([pc.i2w_en[t] for t in top_inds])
    #     print([pc.i2w_en[t] for t in target_inds])
    #
    #     loss += criterion(predictions, sliced_targets)

        # print(t_len)
        # print(len(sliced_targets))
        # print(sliced_targets)
        # if ind == 1:
        #     break
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), 25)
    dc = torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), 25)

    encoder_opt.step()
    decoder_opt.step()

    return loss


def checkpoint_model(encoder, decoder, emb_dim, learning_rate, iterations):
    save_model(encoder, "encoder", emb_dim, learning_rate, iterations)
    save_model(decoder, "decoder", emb_dim*2, learning_rate, iterations)


if __name__ == "__main__":
    batch_size = 32
    position_range = 100
    word_embedding_dim = 500
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

    encoder = EncoderPositionalB(vocab_size=pc.fr_vocab_size,
                                 position_size=position_range,
                                 word_emb_dim=word_embedding_dim,
                                 positional_emb_dim=word_embedding_dim,
                                 use_cuda=use_cuda)
    # encoder = EncoderGRU(input_size=pc.fr_vocab_size,
    #                      hidden_size=word_embedding_dim)

    decoder = DecoderDotAttn(output_size=pc.en_vocab_size,
                             hidden_size=word_embedding_dim*2,
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
        bleu_score = evaluate(encoder, decoder, validation_dataloader, validation_pc)
        bleu_per_iter.append(bleu_score)
        print(f"Bleu Score {iteration+1}: {bleu_score}")

        if bleu_score > best_bleu:
            checkpoint_model(encoder, decoder, word_embedding_dim, learning_rate, iteration)
            best_bleu = bleu_score

        loss_per_iter.append(loss_total)
        loss_total = 0

    save_losses(f"models/loss_training.txt", loss_per_iter)
    save_losses(f"models/bleu_validation.txt", bleu_per_iter)
    save_model(encoder, "encoder", word_embedding_dim, learning_rate, epochs)
    save_model(decoder, "decoder", word_embedding_dim*2, learning_rate, epochs)
