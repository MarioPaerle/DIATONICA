import numpy as np
from scipy.ndimage import gaussian_filter
import collections
import re
from typing import List, Tuple, Dict, Set, Optional


def sinusoidal_encoding(t, d_model):
    """This Function does the positional embedding as described in the "Attention is all you need" paper."
To adapt this to image, d_model can be the square of the side of the image so that the reshaped version
of the encoding vector can be added as input channel.
Pretty sure there's a faster way to do this, but it works.
    """
    encoding = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = 10000 ** (i / d_model)
        encoding[i] = np.sin(t / div_term)
        encoding[i+1] = np.cos(t / div_term)
    return encoding


def add_random_holes(X, p=0.1):
    """This Function adds random 'holes' to the X, with probability (1-p)"""
    mask = np.random.rand(*X.shape) > p 
    return X * mask.astype(X.dtype)


def add_blur(X, sigma=1.0):
    """This Function applies a Gaussian blur to the input array X."""
    return gaussian_filter(X, sigma=sigma)


def timestep_hole(X, t, add_blur=True, hole_per_time=lambda x: 0.1*x, sigma=0.1):
    """This Functions takes X, an image and t, a timestep and adds holes to it.
Then it adds a soft gaussian blur to mimic the effect of convolution networks, and finally
it adds the position encoding channel.

The Probability of holes is computed as a function of the timestep:

X has to be [N, C, W, H]"""
    holed = add_random_holes(X, p=1-hole_per_time(t))
    if add_blur:
        holed = gaussian_filter(holed, sigma=sigma)

    encoded = sinusoidal_encoding(t, X.shape[2]*X.shape[3])
    encoded = encoded.reshape(1, 1, X.shape[2], X.shape[3])
    encoded = np.repeat(encoded, X.shape[0], axis=0)
    return np.concatenate([holed, encoded], axis=1)


def square_hole(X, p, l=1, m=100):
    """
    Generates random squares of zeros inside the given array X with probability p.

    Parameters:
    - X: The input array of shape [N, C, W, H].
    - p: The probability (between 0 and 1) of a square being placed in each channel.

    Returns:
    - An array of the same shape as X, with random squares replaced by zeros.
    """
    N, C, W, H = X.shape
    for n in range(N):
        for c in range(C):
            if np.random.rand() < p:
                square_size = np.random.randint(l, m)
                x_start = np.random.randint(0, W - square_size + 1)
                y_start = np.random.randint(0, H - square_size + 1)
                X[n, c, x_start:x_start + square_size, y_start:y_start + square_size] = 0
    return X


class StringBPE:
    def __init__(self, unknown_token: str = "<UNK>"):
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.unknown_token = unknown_token

    def _get_stats(self, sequences: List[List[str]]) -> collections.defaultdict:
        stats = collections.defaultdict(int)
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                stats[(sequence[i], sequence[i+1])] += 1
        return stats

    def _merge_pair(self, pair_to_merge: Tuple[str, str], sequences_in: List[List[str]]) -> List[List[str]]:
        token_a, token_b = pair_to_merge
        new_token = token_a + ' ' + token_b
        sequences_out = []
        for sequence in sequences_in:
            new_sequence = []
            i = 0
            while i < len(sequence):
                if (i < len(sequence) - 1 and
                        sequence[i] == token_a and sequence[i+1] == token_b):
                    new_sequence.append(new_token)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            sequences_out.append(new_sequence)
        return sequences_out

    def train(self, sequences: List[List[str]], num_merges: int, min_frequency: int = 1) -> None:
        initial_vocab = set()
        for sequence in sequences:
            initial_vocab.update(sequence)
        self.vocab = initial_vocab.copy() # Start with base tokens
        self.merges = []

        current_sequences = [list(seq) for seq in sequences] # Ensure mutability

        print(f"--- Starting BPE Training ---")
        print(f"Initial Vocab Size: {len(self.vocab)}")

        for i in range(num_merges):
            stats = self._get_stats(current_sequences)

            if not stats:
                print("No more pairs to merge.")
                break

            best_pair = max(stats, key=stats.get)
            pair_freq = stats[best_pair]

            if pair_freq < min_frequency:
                print(f"Stopping merge: Frequency of {best_pair} ({pair_freq}) is below threshold ({min_frequency}).")
                break

            print(f"Merge {i+1}/{num_merges}: Merging {best_pair} (frequency: {pair_freq})")

            current_sequences = self._merge_pair(best_pair, current_sequences)

            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)
            self.merges.append(best_pair)

        print("\n--- Training Complete ---")
        print(f"Final Vocab Size: {len(self.vocab)}")
        print(f"Total Merges Performed: {len(self.merges)}")

        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build token_to_id and id_to_token mappings."""
        if self.unknown_token not in self.vocab:
             self.vocab.add(self.unknown_token)

        sorted_vocab = sorted(list(self.vocab))
        self.token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        print(f"Built token-ID mappings. Vocab size (incl. {self.unknown_token}): {len(self.token_to_id)}")

    def tokenize(self, sequence: List[str]) -> List[str]:
        if not self.merges:
            print("Warning: BPE model has not been trained or no merges were learned. Returning original sequence.")
            return list(sequence)

        current_sequence = list(sequence)
        for token_a, token_b in self.merges:
            new_token = token_a + token_b
            i = 0
            next_sequence = []
            while i < len(current_sequence):
                if (i < len(current_sequence) - 1 and
                        current_sequence[i] == token_a and current_sequence[i+1] == token_b):
                    next_sequence.append(new_token)
                    i += 2
                else:
                    next_sequence.append(current_sequence[i])
                    i += 1
            current_sequence = next_sequence

        return current_sequence

    def tokenize_vector(self, sequence: List[str]) -> List[int]:
        if not self.token_to_id:
            raise RuntimeError("BPE model must be trained before tokenizing to vector.")

        tokenized_sequence = self.tokenize(sequence)
        unknown_id = self.token_to_id.get(self.unknown_token)
        if unknown_id is None:
             # This should not happen if _build_mappings worked correctly
             raise RuntimeError(f"Unknown token '{self.unknown_token}' not found in vocabulary mapping.")
        for token in tokenized_sequence:
            if token not in self.token_to_id:
                print(token)
        return [self.token_to_id.get(token, unknown_id) for token in tokenized_sequence]

    def detokenize(self, vector: List[int]) -> List[str]:
        if not self.id_to_token:
             raise RuntimeError("BPE model must be trained before detokenizing.")

        unknown_token_str = self.unknown_token # Use the actual string for unknown IDs
        return [self.id_to_token.get(token_id, unknown_token_str) for token_id in vector]


if __name__ == '__main__':
    corpus = [
        ['15.64.76', '14.15.36', '14.6.3', '15.64.76', '14.15.36'],  # Sequence 1
        ['14.15.36', '14.6.3', '99.99.99'],  # Sequence 2
        ['15.64.76', '14.15.36']  # Sequence 3
    ]

    num_merges = 3
    bpe_processor = StringBPE(unknown_token="<UNK>")
    bpe_processor.train(corpus, num_merges)

    print("\nLearned Merges:", bpe_processor.merges)

    sequence_to_tokenize = ['15.100.76', '14.15.36', '14.6.3', '99.99.99', '15.64.76', 'unknown_token']
    tokenized_strings = bpe_processor.tokenize(sequence_to_tokenize)
    print(f"Tokenized (Strings): {tokenized_strings}")
    tokenized_vector = bpe_processor.tokenize_vector(sequence_to_tokenize)

    vector_to_detokenize = tokenized_vector + [100]
    print(f"Vector to Detokenize: {vector_to_detokenize}")

    detokenized_sequence = bpe_processor.detokenize(vector_to_detokenize)
    print(f"Detokenized Sequence: {detokenized_sequence}")

    user_sequence = ['15.64.76', '14.15.36', '14.6.3']
    tokenized_user_strings = bpe_processor.tokenize(user_sequence)
    tokenized_user_vector = bpe_processor.tokenize_vector(user_sequence)
    detokenized_user_sequence = bpe_processor.detokenize(tokenized_user_vector)
