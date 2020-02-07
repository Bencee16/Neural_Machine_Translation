import spacy


def tokenize_sentence(sentence, lm=None):
    """
    Tokenize a single sentence
    """
    return lm(sentence)


def french_tokenizer():
    french_nlp = spacy.load("fr")
    return lambda sent: tokenize_sentence(sent, lm=french_nlp)


def english_tokenizer():
    english_nlp = spacy.load("en")
    return lambda sent: tokenize_sentence(sent, lm=english_nlp)


if __name__ == "__main__":
    e_tokenize = english_tokenizer()
    print(list(e_tokenize("Two young, White males are outside near many bushes.")))
    f_tokenize = french_tokenizer()
    print(list(f_tokenize("Deux jeunes hommes blancs sont dehors près de buissons.")))
