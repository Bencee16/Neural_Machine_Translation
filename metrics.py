import torch
from data_loader_batch import FrEnParallelCorpusDataset
from nltk.translate.bleu_score import sentence_bleu


def generate_prediction(encoder_model, decoder_model, input_sentence, i2w, max_length=40):
    with torch.no_grad():
        encoder_model.train(False)
        decoder_model.train(False)
        encoder_output, emb_means = encoder_model(input_sentence, [len(input_sentence)])
        # encoder_output, hidden = encoder_model(input_sentence, [len(input_sentence)])
        decoder_input = torch.tensor([[FrEnParallelCorpusDataset.start_token_ind]],
                                     device=encoder_output.device)
        decoder_hidden = emb_means.unsqueeze(0)
        # decoder_hidden = hidden[:2]

        decoded_tokens = []
        for di in range(max_length):
            decoder_output, decoder_hidden, attn = decoder_model(decoder_input,
                                                           decoder_hidden ,encoder_output)

            topv, topi = decoder_output.topk(1)
            decoded_tokens.append(i2w[topi.item()])
            if topi.item() == FrEnParallelCorpusDataset.stop_token_ind:
                break

            decoder_input = topi.squeeze(1).detach()

        encoder_model.train(True)
        decoder_model.train(True)
        return decoded_tokens


def generate_prediction_gru(encoder_model, decoder_model, input_sentence, i2w, max_length=40):
    with torch.no_grad():
        encoder_model.train(False)
        decoder_model.train(False)
        encoder_output, hidden = encoder_model(input_sentence, [len(input_sentence)])
        decoder_input = torch.tensor([[FrEnParallelCorpusDataset.start_token_ind]],
                                     device=encoder_output.device)
        decoder_hidden = hidden[:1]

        decoded_tokens = []
        for di in range(max_length):
            decoder_output, decoder_hidden, attn = decoder_model(decoder_input,
                                                           decoder_hidden ,encoder_output)

            topv, topi = decoder_output.topk(1)
            decoded_tokens.append(i2w[topi.item()])
            if topi.item() == FrEnParallelCorpusDataset.stop_token_ind:
                break

            decoder_input = topi.squeeze(1).detach()

        encoder_model.train(True)
        decoder_model.train(True)
        return decoded_tokens


def generate_prediction_gru_multi(encoder_model, decoder_model, input_sentence, i2w, max_length=40):
    with torch.no_grad():
        encoder_model.train(False)
        decoder_model.train(False)
        encoder_output, hidden = encoder_model(input_sentence, [len(input_sentence)])
        decoder_input = torch.tensor([[FrEnParallelCorpusDataset.start_token_ind]],
                                     device=encoder_output.device)
        decoder_hidden = hidden[:encoder_model.n_layers]

        decoded_tokens = []
        for di in range(max_length):
            decoder_output, decoder_hidden, attn = decoder_model(decoder_input,
                                                           decoder_hidden ,encoder_output)

            topv, topi = decoder_output.topk(1)
            decoded_tokens.append(i2w[topi.item()])
            if topi.item() == FrEnParallelCorpusDataset.stop_token_ind:
                break

            decoder_input = topi.squeeze(1).detach()

        encoder_model.train(True)
        decoder_model.train(True)
        return decoded_tokens


def index_tensor_to_tokens(index_tensor, corpus, english=True):
    if english:
        i2w = corpus.i2w_en
    else:
        i2w = corpus.i2w_fr
    return [i2w[token] for token in index_tensor.cpu().numpy().squeeze()]


def evaluate(encoder, decoder, dataloader, corpus, metrics=None, max_length=40):
    if metrics is None:
        metrics = []
    count = 0
    bleu_sum = 0

    for ind, batch_data in enumerate(dataloader):
        source_tensor, source_lens, target_tensor, target_lens = batch_data
        tokens = generate_prediction(encoder, decoder, source_tensor, corpus.i2w_en)
        target_tokens = index_tensor_to_tokens(target_tensor, corpus)
        count += 1
        bleu_sum += bleu_score(tokens, target_tokens)
        if ind % 1000 == 0:
            print(tokens)
            print([corpus.i2w_fr[token] for token in source_tensor.cpu().numpy().squeeze()])

    return bleu_sum/count


def evaluate_gru(encoder, decoder, dataloader, corpus, metrics=None, max_length=40):
    if metrics is None:
        metrics = []
    count = 0
    bleu_sum = 0

    for ind, batch_data in enumerate(dataloader):
        source_tensor, source_lens, target_tensor, target_lens = batch_data
        tokens = generate_prediction_gru(encoder, decoder, source_tensor, corpus.i2w_en)
        target_tokens = index_tensor_to_tokens(target_tensor, corpus)
        count += 1
        bleu_sum += bleu_score(tokens, target_tokens)
        if ind % 1000 == 0:
            print(tokens)
            print([corpus.i2w_fr[token] for token in source_tensor.cpu().numpy().squeeze()])

    return bleu_sum/count


def evaluate_gru_multi(encoder, decoder, dataloader, corpus, metrics=None, max_length=40):
    if metrics is None:
        metrics = []
    count = 0
    bleu_sum = 0

    for ind, batch_data in enumerate(dataloader):
        source_tensor, source_lens, target_tensor, target_lens = batch_data
        tokens = generate_prediction_gru_multi(encoder, decoder, source_tensor, corpus.i2w_en)
        target_tokens = index_tensor_to_tokens(target_tensor, corpus)
        count += 1
        bleu_sum += bleu_score(tokens, target_tokens)
        if ind % 1000 == 0:
            print(tokens)
            print([corpus.i2w_fr[token] for token in source_tensor.cpu().numpy().squeeze()])

    return bleu_sum/count


def join_tokens(sentence):
    output = []
    for token in sentence:
        if len(output) > 0:
            if output[-1].endswith("@@"):
                output[-1] = output[-1][:-2] + token
            else:
                output.append(token)
        else:
            output.append(token)

    return output


def bleu_score(predicted_tokens, target_tokens):
    bleu_score = sentence_bleu([join_tokens(target_tokens)], join_tokens(predicted_tokens))
    return bleu_score


def file_to_ter_format(filename, output_name):
    with open(filename, "r") as file:
        with open(output_name, "w") as ter_file:
            line = file.readline()
            count = 0
            while line:
                line = line.strip()
                ter_file.write(f"{line} ({count})\n")
                count += 1
                line = file.readline()


if __name__ == "__main__":
    print(bleu_score(["this", "is", "a", "test"], ["this", "is", "a", "test"]))
    file_to_ter_format("./data/test_e_tokenized", "./data/test_ter.en")

