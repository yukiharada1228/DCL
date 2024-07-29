import math

import torch


class SingleGPURASampler(torch.utils.data.Sampler):
    """
    Sampler that implements repeated augmentation for a single GPU.
    This sampler repeatedly samples from the dataset, allowing for
    multiple augmented versions of each sample in a single epoch.
    """

    def __init__(self, dataset, shuffle=True, num_repeats=3):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0

        # Calculate the total number of samples after repetition
        self.num_samples = len(self.dataset) * self.num_repeats

        # Adjust num_samples to be divisible by a common batch size (e.g., 256)
        self.num_selected_samples = int(math.floor(self.num_samples // 256 * 256))

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Repeat indices
        indices = [index for index in indices for _ in range(self.num_repeats)]

        # Add extra samples to make it evenly divisible if necessary
        if len(indices) < self.num_selected_samples:
            extras = self.num_selected_samples - len(indices)
            indices += indices[:extras]
        else:
            indices = indices[: self.num_selected_samples]

        assert len(indices) == self.num_selected_samples

        return iter(indices)

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
