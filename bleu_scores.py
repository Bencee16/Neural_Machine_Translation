from nltk.translate.bleu_score import corpus_bleu

def read_ter_file(filename, is_ref=True):
    token_lists = []
    with open(filename, "r") as ter_file:
        line = ter_file.readline()
        while line:
            # get tokens and remove number
            tokens = line.strip().split(" ")[:-1]
            if is_ref:
                tokens = [tokens]
            token_lists.append(tokens)
            line = ter_file.readline()
    return token_lists


def compute_bleu_score_corpus(refs, predictions):
    print('Cumulative 1-gram: %f' % corpus_bleu(refs, predictions, weights=(1, 0, 0, 0)))
    print('Cumulative 2-gram: %f' % corpus_bleu(refs, predictions, weights=(0.5, 0.5, 0, 0)))
    print('Cumulative 3-gram: %f' % corpus_bleu(refs, predictions, weights=(0.33, 0.33, 0.33, 0)))
    print('Cumulative 4-gram: %f' % corpus_bleu(refs, predictions, weights=(0.25, 0.25, 0.25, 0.25)))


def bleu_scores(preds_file, ref_file="./data/test_ter.en"):
    refs = read_ter_file(ref_file)
    print(refs)
    preds = read_ter_file(preds_file, is_ref=False)
    compute_bleu_score_corpus(refs, preds)


if __name__ == "__main__":
    print("Positional")
    bleu_scores("./predictions/predictions_500.en")

    print("GRU")
    bleu_scores("./predictions/predictions_gru_enc_1000.en")

    print("Multi")
    bleu_scores("./predictions/predictions_gru_enc_multi_1000_1000.en")
