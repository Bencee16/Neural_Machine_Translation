import subprocess
from collections import Counter
from preprocessing.tokenize import english_tokenizer, french_tokenizer
from preprocessing.lowercase import lowercase_tokens
import subword_nmt.learn_bpe


def extract_vocabulary_and_tokenized_lines(filename, tokenizer):
    tokenized_lines = list()
    with open(filename, "r") as file:
        line = file.readline()
        cnt = 1
        vocab_counter = Counter()

        while line:
            if cnt % 1000 is 0:
                print(f"Line: {cnt}")

            tokens = tokenizer(line.strip())
            formatted_tokens = lowercase_tokens(tokens)
            tokenized_lines.append(formatted_tokens)
            vocab_counter.update(formatted_tokens)

            cnt += 1
            line = file.readline()

    return vocab_counter, tokenized_lines


def write_tokenized_lines_to_file(lines, outfile):
    with open(outfile, "w") as outbuf:
        for line in lines:
            token_line = " ".join(line)
            outbuf.write(f"{token_line}\n")


def write_vocab_to_file(vocab_counter, outfile):
    with open(outfile, "w") as outbuf:
        for (token, count) in vocab_counter.items():
            if token != " ":
                outbuf.write(f"{token} {count}\n")


def generate_and_apply_bpe(filename, tokenizer, tokenized_file, vocab_file, bpe_file, num_symbols):
    print("Extracting vocab and tokenizing")
    vocab_counter, tokenized_lines = extract_vocabulary_and_tokenized_lines(filename, tokenizer)
    print("Finished extraction, writing tokenized file...")
    write_tokenized_lines_to_file(tokenized_lines, tokenized_file)
    print("Writing vocab file...")
    write_vocab_to_file(vocab_counter, vocab_file)
    print("Finished writing...")
    print("Generating BPE")
    subword_nmt.learn_bpe.learn_bpe(infile=open(vocab_file, 'r'),
                                    outfile=open(bpe_file, 'w'),
                                    num_symbols=num_symbols,
                                    is_dict=True)


def en_generate_and_apply_bpe(filename, tokenized_file, vocab_file, bpe_file, num_symbols=5000):
    tokenizer = english_tokenizer()
    generate_and_apply_bpe(filename=filename,
                           tokenizer=tokenizer,
                           tokenized_file=tokenized_file,
                           vocab_file=vocab_file,
                           bpe_file=bpe_file,
                           num_symbols=num_symbols)


def fr_generate_and_apply_bpe(filename, tokenized_file, vocab_file, bpe_file, num_symbols=5000):
    tokenizer = french_tokenizer()
    generate_and_apply_bpe(filename=filename,
                           tokenizer=tokenizer,
                           tokenized_file=tokenized_file,
                           vocab_file=vocab_file,
                           bpe_file=bpe_file,
                           num_symbols=num_symbols)


def tokenize_file(tokenizer, input_file, output_file):
    _, tokenized_lines = extract_vocabulary_and_tokenized_lines(input_file, tokenizer)
    write_tokenized_lines_to_file(tokenized_lines, output_file)


def tokenize_file_en(input_file, output_file):
    tokenizer = english_tokenizer()
    tokenize_file(tokenizer, input_file, output_file)


def tokenize_file_fr(input_file, output_file):
    tokenizer = french_tokenizer()
    tokenize_file(tokenizer, input_file, output_file)


if __name__ == "__main__":
    # NOTE: CLONE https://github.com/rsennrich/subword-nmt into project 2 directory
    print("Starting with english...")
    # First English
    e_filename = "../data/train.en"
    e_token_filename = "../data/train_e_tokenized"
    e_token_vocab_filename = "../data/train_e_vocab"
    e_bpe_filename = "../data/train_e_bpe_voc"
    en_generate_and_apply_bpe(filename=e_filename,
                              tokenized_file=e_token_filename,
                              vocab_file=e_token_vocab_filename,
                              bpe_file=e_bpe_filename)
    subprocess.call(f"../subword-nmt/subword_nmt/apply_bpe.py -c {e_bpe_filename} < {e_token_filename} > ../data/train_e_bpe", shell=True)
    # also validation
    print("English validation set...")
    e_val_token_filename = "../data/val_e_tokenized"
    tokenize_file_en("../data/val.en", e_val_token_filename)
    subprocess.call(f"../subword-nmt/subword_nmt/apply_bpe.py -c {e_bpe_filename} < {e_val_token_filename} > ../data/val_e_bpe", shell=True)
    # and test
    print("English test set...")
    e_test_token_filename = "../data/test_e_tokenized"
    tokenize_file_en("../data/test_2017_flickr.en", e_test_token_filename)
    subprocess.call(f"../subword-nmt/subword_nmt/apply_bpe.py -c {e_bpe_filename} < {e_test_token_filename} > ../data/test_e_bpe", shell=True)

    # Then French
    print("Next up is french...")
    f_filename = "../data/train.fr"
    f_token_filename = "../data/train_f_tokenized"
    f_token_vocab_filename = "../data/train_f_vocab"
    f_bpe_filename = "../data/train_f_bpe_voc"
    fr_generate_and_apply_bpe(filename=f_filename,
                              tokenized_file=f_token_filename,
                              vocab_file=f_token_vocab_filename,
                              bpe_file=f_bpe_filename)
    subprocess.call(f"../subword-nmt/subword_nmt/apply_bpe.py -c {f_bpe_filename} < {f_token_filename} > ../data/train_f_bpe", shell=True)
    # also validation
    print("French validation set...")
    f_val_token_filename = "../data/val_f_tokenized"
    tokenize_file_fr("../data/val.fr", f_val_token_filename)
    subprocess.call(f"../subword-nmt/subword_nmt/apply_bpe.py -c {f_bpe_filename} < {f_val_token_filename} > ../data/val_f_bpe", shell=True)
    # and test
    print("French test set...")
    f_test_token_filename = "../data/test_f_tokenized"
    tokenize_file_fr("../data/test_2017_flickr.fr", f_test_token_filename)
    subprocess.call(f"../subword-nmt/subword_nmt/apply_bpe.py -c {f_bpe_filename} < {f_test_token_filename} > ../data/test_f_bpe", shell=True)
