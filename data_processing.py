import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os


class Dictionary:
    """
    Dictionary class that maps each word to a unique token index.
    """
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.nb_occ = {"<PAD>": 0, "<UNK>": 0, "<BOS>": 0, "<EOS>": 0}
    
    def add_word(self, word):
        """
        Adds a word to the dictionary if it doesn't exist.
        Increments occurrence counter for the word.
        Returns the token ID.
        """
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.nb_occ[word] = 1
        else:
            self.nb_occ[word] += 1
        
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)


class CorpusWords:
    """
    Corpus class that processes text files and converts them to token sequences.
    """
    def __init__(self, path):
        self.dictionary = Dictionary()
        
        # First pass: build vocabulary from training file
        self.build_vocab(os.path.join(path, 'train.txt'))
        
        # Second pass: tokenize all files
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def build_vocab(self, path):
        """Builds vocabulary from training file only"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                words = ["<BOS>"] + words + ["<EOS>"]
                for word in words:
                    self.dictionary.add_word(word)
        

    def tokenize(self, path):
        """
        Tokenizes a text file and returns a LongTensor of token indices.
        """
        tokens = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                tokens.append(self.dictionary.word2idx["<BOS>"])
                for word in words:
                    if word in self.dictionary.word2idx:
                        tokens.append(self.dictionary.word2idx[word])
                    else:
                        tokens.append(self.dictionary.word2idx["<UNK>"])
                tokens.append(self.dictionary.word2idx["<EOS>"])
        
        return torch.LongTensor(tokens)
    
    def get_vocabulary_size(self):
        return len(self.dictionary)
    
    def decode(self, tensor):
        """
        Converts a tensor of token indices back to words.
        """
        words = []
        for idx in tensor:
            idx_val = idx.item() if hasattr(idx, 'item') else idx
            if idx_val < len(self.dictionary.idx2word):
                word = self.dictionary.idx2word[idx_val]
                if word not in ["<PAD>", "<BOS>", "<EOS>"]:
                    words.append(word)
            else:
                words.append("<UNK>")
        return " ".join(words)


def get_batches(data, batch_size, seq_length, device='cpu'):
    """
    Organizes data into batches for training.
    """
    # Total number of batches
    n_batches = data.size(0) // (batch_size * seq_length)
    # Trim data to fit evenly into batches, we need to remove the last part of the data that doesn't fit into a batch
    data = data[:n_batches * batch_size * seq_length]
    # Reshape data to (batch_size, n_batches * seq_length)
    data = data.view(batch_size, -1) 
    num_batches = (data.size(1) - seq_length) // seq_length
    
    all_inputs = torch.zeros(num_batches, batch_size, seq_length, dtype=torch.long, device=device)
    all_targets = torch.zeros(num_batches, batch_size, seq_length, dtype=torch.long, device=device)
    
    # Fill tensors with batches
    for batch_idx, i in enumerate(range(0, data.size(1) - seq_length, seq_length)):
        if i + seq_length + 1 <= data.size(1):
            all_inputs[batch_idx] = data[:, i:i+seq_length]
            all_targets[batch_idx] = data[:, i+1:i+seq_length+1]
    
    return all_inputs.to(device), all_targets.to(device)


def main(data_path="wikitext2", batch_size=32, seq_length=35, device='cpu', verbose=False):
    """
    Main function to process the data.
    """
    dataset = CorpusWords(data_path)
    
    if verbose:
        print(f"Vocabulary size: {dataset.get_vocabulary_size()}")
    
    # Get raw token data
    train_data = dataset.train
    valid_data = dataset.valid
    test_data = dataset.test
    
    if verbose:
        print(f"Train data size: {train_data.size()}")
        print(f"Validation data size: {valid_data.size()}")
        print(f"Test data size: {test_data.size()}")
    
    # Create batches
    train_inputs, train_targets = get_batches(train_data, batch_size=batch_size, 
                                             seq_length=seq_length, device=device)
    
    val_inputs, val_targets = get_batches(valid_data, batch_size=batch_size, 
                                         seq_length=seq_length, device=device)
    
    test_inputs, test_targets = get_batches(test_data, batch_size=batch_size, 
                                           seq_length=seq_length, device=device)
    
    if verbose:
        print(f"Train inputs shape: {train_inputs.shape}")
    
    return dataset, train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset, train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = main(
        data_path="wikitext2",
        batch_size=32, 
        seq_length=35,
        device=device,
        verbose=True  # Set to False to disable all prints
    )