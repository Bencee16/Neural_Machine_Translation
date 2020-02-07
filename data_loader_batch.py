import os
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class SubsetSampler(Sampler):
    def __init__(self, number_of_items):
        self.number_of_items = number_of_items

    def __iter__(self):
        return iter(range(self.number_of_items))

    def __len__(self):
        return self.number_of_items


class FrEnParallelCorpusDataset(Dataset):
    pad_token = "PAD"
    stop_token_ind = 0
    start_token_ind = 1

    def __init__(self, source_fr_training_file, target_en_training_file,
                 other_fr_files=None, other_en_files=None,
                 cached_data_filename="./data/training_dataset.pkl",
                 use_cached_training_tuples=True,
                 use_cuda=False):
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda_device = torch.device("cuda")

        self.cached_data_filename = cached_data_filename
        self.source_f_fname = source_fr_training_file
        self.target_e_fname = target_en_training_file
        self.pad_token = FrEnParallelCorpusDataset.pad_token
        if other_en_files is not None:
            self.other_fr_files = other_fr_files
            self.other_en_files = other_en_files
        else:
            self.other_fr_files = []
            self.other_en_files = []

        if os.path.isfile(self.cached_data_filename):
            self.load_cached_data(use_cached_training_tuples)
        else:
            self.w2i_fr = {}
            self.i2w_fr = {}
            self.w2i_en = {}
            self.i2w_en = {}
            self.fr_vocab_size = 0
            self.en_vocab_size = 0
            self.training_tuples = []
            self.build_parallel_corpus()

    def __len__(self):
        return len(self.training_tuples)

    def __getitem__(self, item_ind):
        """Return the indices of the french and english tokens"""
        fr_tokens, en_tokens = self.training_tuples[item_ind]

        # Add EOS token if not present
        if "." not in fr_tokens:
            fr_tokens.append(".")
        if "." not in en_tokens:
            en_tokens.append(".")

        fr_inds = [self.w2i_fr[ft] for ft in fr_tokens]
        en_inds = [self.w2i_en[et] for et in en_tokens]

        return fr_inds, en_inds

    def build_parallel_corpus(self):
        """Build the parallel corpus from the preprocessed data"""
        print("Building parallel corpus...")

        # Add stop tokens
        stop_token = "."
        start_token = "SOS"
        self.w2i_fr[stop_token] = 0
        self.w2i_en[stop_token] = 0
        self.i2w_fr[0] = stop_token
        self.i2w_en[0] = stop_token
        self.w2i_fr[start_token] = 1
        self.w2i_en[start_token] = 1
        self.i2w_fr[1] = start_token
        self.i2w_en[1] = start_token
        n_fr_words = 2
        n_en_words = 2

        with open(self.source_f_fname) as fr_file:
            with open(self.target_e_fname) as en_file:
                fr_line = fr_file.readline()
                en_line = en_file.readline()
                while fr_line:
                    fr_tokens = fr_line.strip().split(" ")
                    en_tokens = en_line.strip().split(" ")

                    # Build vocabularies
                    for ft in fr_tokens:
                        if ft not in self.w2i_fr:
                            self.w2i_fr[ft] = n_fr_words
                            self.i2w_fr[n_fr_words] = ft
                            n_fr_words += 1

                    for et in en_tokens:
                        if et not in self.w2i_en:
                            self.w2i_en[et] = n_en_words
                            self.i2w_en[n_en_words] = et
                            n_en_words += 1

                    self.training_tuples.append((fr_tokens, en_tokens))

                    # Grab next
                    fr_line = fr_file.readline()
                    en_line = en_file.readline()

        for f_name in self.other_fr_files:
            with open(f_name) as fr_file:
                fr_line = fr_file.readline()
                while fr_line:
                    fr_tokens = fr_line.strip().split(" ")
                    # Build vocabularies
                    for ft in fr_tokens:
                        if ft not in self.w2i_fr:
                            self.w2i_fr[ft] = n_fr_words
                            self.i2w_fr[n_fr_words] = ft
                            n_fr_words += 1
                    fr_line = fr_file.readline()

        for en_name in self.other_en_files:
            with open(en_name) as en_file:
                en_line = en_file.readline()
                while en_line:
                    en_tokens = en_line.strip().split(" ")
                    # Build vocabularies
                    for et in en_tokens:
                        if et not in self.w2i_en:
                            self.w2i_en[et] = n_en_words
                            self.i2w_en[n_en_words] = et
                            n_en_words += 1
                    en_line = en_file.readline()

        self.fr_vocab_size = n_fr_words
        self.en_vocab_size = n_en_words
        self.cache_data()

    def load_cached_data(self, load_cached_tuples):
        print("Loading cached parallel corpus...")
        with open(self.cached_data_filename, "rb") as cache_file:
            data_dict = pickle.load(cache_file)
            self.w2i_fr = data_dict["w2i_fr"]
            self.w2i_en = data_dict["w2i_en"]
            self.i2w_fr = data_dict["i2w_fr"]
            self.i2w_en = data_dict["i2w_en"]
            self.fr_vocab_size = data_dict["fr_vocab_size"]
            self.en_vocab_size = data_dict["en_vocab_size"]
            self.training_tuples = data_dict["training_tuples"]

        if not load_cached_tuples:
            self.load_training_tuples()

    def load_training_tuples(self):
        self.training_tuples = []

        with open(self.source_f_fname) as fr_file:
            with open(self.target_e_fname) as en_file:
                fr_line = fr_file.readline()
                en_line = en_file.readline()
                while fr_line:
                    fr_tokens = fr_line.strip().split(" ")
                    en_tokens = en_line.strip().split(" ")

                    self.training_tuples.append((fr_tokens, en_tokens))

                    # Grab next
                    fr_line = fr_file.readline()
                    en_line = en_file.readline()

    def cache_data(self):
        print("Caching parallel corpus...")
        data_dict = dict()
        data_dict["w2i_fr"] = self.w2i_fr
        data_dict["w2i_en"] = self.w2i_en
        data_dict["i2w_fr"] = self.i2w_fr
        data_dict["i2w_en"] = self.i2w_en
        data_dict["fr_vocab_size"] = self.fr_vocab_size
        data_dict["en_vocab_size"] = self.en_vocab_size
        data_dict["training_tuples"] = self.training_tuples

        with open(self.cached_data_filename, "wb") as cache_file:
            pickle.dump(data_dict, cache_file, pickle.HIGHEST_PROTOCOL)


def pad_tokens(token_list, total_length):
    token_list += [1 for _ in range(total_length - len(token_list))]
    return token_list


def collate_fn_cuda(use_cuda):
    def collate_fn(data):
        source_data = []
        target_data = []

        for s_inds, t_inds in data:
            source_data.append(s_inds)
            target_data.append(t_inds)

        data_zipped = sorted(zip(source_data, target_data), key=lambda s: len(s[0]), reverse=True)
        sorted_source, sorted_target = zip(*data_zipped)

        # Pad the data
        source_lens = [len(s) for s in sorted_source]
        padded_source = [pad_tokens(s, max(source_lens)) for s in sorted_source]
        target_lens = [len(t) for t in sorted_target]
        padded_target = [pad_tokens(t, max(target_lens)) for t in sorted_target]

        # dim S x B
        source_tensor = torch.tensor(padded_source, dtype=torch.int64).transpose(0, 1)
        # dim T x B
        target_tensor = torch.tensor(padded_target, dtype=torch.int64).transpose(0, 1)

        if use_cuda:
            source_tensor = source_tensor.cuda()
            target_tensor = target_tensor.cuda()

        return source_tensor, source_lens, target_tensor, target_lens

    return collate_fn


if __name__ == "__main__":
    pc = FrEnParallelCorpusDataset("./data/train_f_bpe", "./data/train_e_bpe")
    batch_size = 32
    from torch.utils.data import DataLoader
    training_dataloader = DataLoader(dataset=pc,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn_cuda(True),
                                     shuffle=True)
    for batch in training_dataloader:
        break
