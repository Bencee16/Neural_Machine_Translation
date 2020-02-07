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

    def __init__(self, source_fr_training_file, target_en_training_file,
                 cached_data_filename="./data/training_dataset.pkl",
                 use_cuda=False):
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda_device = torch.device("cuda")

        self.cached_data_filename = cached_data_filename
        self.source_f_fname = source_fr_training_file
        self.target_e_fname = target_en_training_file
        self.pad_token = FrEnParallelCorpusDataset.pad_token

        if os.path.isfile(self.cached_data_filename):
            self.load_cached_data()
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

        if self.use_cuda:
            return torch.tensor(fr_inds, device=self.cuda_device), torch.tensor(en_inds, device=self.cuda_device)
        return torch.tensor(fr_inds), torch.tensor(en_inds)

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
        self.w2i_fr[self.pad_token] = 2
        self.w2i_en[self.pad_token] = 2
        self.i2w_fr[2] = self.pad_token
        self.i2w_en[2] = self.pad_token
        n_fr_words = 3
        n_en_words = 3

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

        self.fr_vocab_size = n_fr_words
        self.en_vocab_size = n_en_words
        self.cache_data()

    def load_cached_data(self):
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


if __name__ == "__main__":
    pc = FrEnParallelCorpusDataset("./data/train_f_bpe", "./data/train_e_bpe")
