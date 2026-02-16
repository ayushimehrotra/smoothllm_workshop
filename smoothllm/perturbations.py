import random
import string


class Perturbation:
    """Base class for random perturbations.

    Attributes:
        q: Perturbation budget. Semantics vary by subclass:
            - RandomSwapPerturbation: absolute number of characters to swap.
            - RandomPatchPerturbation: absolute number of contiguous characters
              to replace.
            - RandomInsertPerturbation: percentage of string length to insert
              (e.g., q=10 inserts ~10% extra characters).
        alphabet: Character set used for replacements/insertions.
    """

    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable


class RandomSwapPerturbation(Perturbation):
    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2.

    ``q`` is the absolute number of characters to swap with random replacements.
    """

    def __init__(self, q):
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        num_swaps = min(self.q, len(s))
        sampled_indices = random.sample(range(len(s)), num_swaps)
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return "".join(list_s)


class RandomPatchPerturbation(Perturbation):
    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2.

    ``q`` is the absolute number of contiguous characters to replace.
    """

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        substring_width = min(self.q, len(s))
        if substring_width == 0:
            return s
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = "".join(
            [random.choice(self.alphabet) for _ in range(substring_width)]
        )
        list_s[start_index : start_index + substring_width] = sampled_chars
        return "".join(list_s)


class RandomInsertPerturbation(Perturbation):
    """Implementation of random insert perturbations.
    See `RandomInsertPerturbation` in lines 11-17 of Algorithm 2.

    ``q`` is a percentage: the number of characters inserted equals
    ``int(len(s) * q / 100)``.
    """

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        num_insertions = int(len(s) * self.q / 100)
        if num_insertions == 0:
            return s
        sampled_indices = sorted(random.sample(range(len(s)), num_insertions))
        for offset, i in enumerate(sampled_indices):
            list_s.insert(i + offset, random.choice(self.alphabet))
        return "".join(list_s)
