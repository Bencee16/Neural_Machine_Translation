import torch
import numpy as np
import model_utils
from metrics import join_tokens


def load_encoder_and_decoder(enc_model, dec_model,
                             embedding_dim, learning_rate,
                             epoch, subscript_name=None):
    if subscript_name is None or "":
        subscript_name = ""
    else:
        subscript_name = f"_{subscript_name}"

    enc_dict = model_utils.load_model_dict(
        model_name=f"encoder{subscript_name}",
        embedding_space=embedding_dim,
        learning_rate=learning_rate,
        iterations=epoch
    )
    dec_dict = model_utils.load_model_dict(
        model_name=f"decoder{subscript_name}",
        embedding_space=embedding_dim*2,
        learning_rate=learning_rate,
        iterations=epoch
    )
    enc_model.load_state_dict(enc_dict)
    dec_model.load_state_dict(dec_dict)

    return enc_model, dec_model


def generate_prediction(encoder_model, decoder_model, enc_dec_transform_fn,
                        input_sentence, i2w, max_length=40):
    with torch.no_grad():
        encoder_model.train(False)
        decoder_model.train(False)
        encoder_output, h_out = encoder_model(input_sentence, [len(input_sentence)])
        decoder_input = torch.tensor([[FrEnParallelCorpusDataset.start_token_ind]],
                                     device=encoder_output.device)
        decoder_hidden = enc_dec_transform_fn(h_out)

        decoded_tokens = []
        # decoder_alphas = torch.zeros((max_length, max_length), device=encoder_output.device)
        for di in range(max_length):
            decoder_output, decoder_hidden, attn = decoder_model(decoder_input,
                                                           decoder_hidden ,encoder_output)
            # decoder_alphas[di, :attn.size(1)] = attn.squeeze()
            topv, topi = decoder_output.topk(1)
            decoded_tokens.append(i2w[topi.item()])
            if topi.item() == FrEnParallelCorpusDataset.stop_token_ind:
                break

            decoder_input = topi.squeeze(1).detach()

        encoder_model.train(True)
        decoder_model.train(True)
        return decoded_tokens


def generate_predictions(encoder, decoder,
                         enc_dec_transform_fn, dataloader, corpus):
    predictions = []

    for ind, batch_data in enumerate(dataloader):
        source_tensor, source_lens, target_tensor, target_lens = batch_data
        tokens = generate_prediction(encoder, decoder, enc_dec_transform_fn,
                                     source_tensor, corpus.i2w_en)
        # if ind == 13:
            # np_alphas = alphas.cpu().numpy()
            # print(alphas.cpu().numpy())
            # print(tokens)
            # np.save("./predictions/alphas_multi", np_alphas)
            # print([corpus.i2w_fr[token] for token in source_tensor.cpu().numpy().squeeze()])
        predictions.append(join_tokens(tokens))
    return predictions


def save_predictions_to_file(predictions, emb_dim, subscript=None):
    if subscript is None or "":
        subscript = ""
    else:
        subscript = f"_{subscript}"
    filename = f"./predictions/predictions{subscript}_{emb_dim}.en"

    with open(filename, "w") as file:
        for ind, pred_line in enumerate(predictions):
            pred_sent = " ".join(pred_line)
            file.write(f"{pred_sent} ({ind})\n")


if __name__ == "__main__":
    from networks.EncoderPositional import EncoderPositionalB
    from networks.EncoderGRU import EncoderGRU
    from networks.EncoderGRUMulti import EncoderGRU as EncoderMulti
    from networks.DecoderDotAttn import DecoderDotAttn
    from networks.DecoderDotAttnMulti import DecoderDotAttn as DecoderMulti
    from data_loader_batch import FrEnParallelCorpusDataset, collate_fn_cuda
    from torch.utils.data import DataLoader

    use_cuda = torch.cuda.is_available()
    pc = FrEnParallelCorpusDataset("./data/train_f_bpe", "./data/train_e_bpe",
                                   use_cuda=use_cuda)

    emb_dim = 500
    subscript = "gru_enc"
    subscript = ""
    enc = EncoderPositionalB(pc.fr_vocab_size, 100, emb_dim, emb_dim,
                             use_cuda=use_cuda)
    # enc = EncoderGRU(pc.fr_vocab_size, emb_dim)
    dec = DecoderDotAttn(emb_dim*2, pc.en_vocab_size, use_cuda=use_cuda)

    if use_cuda:
        enc.cuda()
        dec.cuda()

    enc, dec = load_encoder_and_decoder(
        enc_model=enc,
        dec_model=dec,
        embedding_dim=emb_dim,
        learning_rate=1e-4,
        epoch=29
    )

    def transform(enc_hidden):
        return enc_hidden.unsqueeze(0)

    # def transform(enc_hidden):
    #     return enc_hidden[:1]

    # def transform(enc_hidden):
    #     return enc_hidden[:enc.n_layers]


    test_pc = FrEnParallelCorpusDataset("./data/test_f_bpe", "./data/test_e_bpe",
                                              use_cached_training_tuples=False,
                                              use_cuda=use_cuda)

    test_dataloader = DataLoader(dataset=test_pc,
                                       batch_size=1,
                                       collate_fn=collate_fn_cuda(use_cuda),
                                       shuffle=False)

    preds = generate_predictions(enc, dec, transform, test_dataloader, pc)
    save_predictions_to_file(preds, emb_dim)
