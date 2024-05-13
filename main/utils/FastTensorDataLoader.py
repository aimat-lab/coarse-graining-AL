import torch


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, drop_last=True):
        """
        Initialize a FastTensorDataLoader.

        *tensors (torch.Tensor): tensors to store. Must have the same length @ dim 0.
        batch_size (int): batch size to load.
        shuffle (bool): if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        drop_last (bool): if True, drop the last incomplete batch, if the dataset
            size is not divisible by the batch size. If False and the size of dataset
            is not divisible by the batch size, then the last batch will be smaller.
            Defaults to True.
        """

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0 and not drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len or (
            self.drop_last and (self.dataset_len - self.i) < self.batch_size
        ):
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
