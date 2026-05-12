import torch


class Compose:
    def __init__(self, transforms, generator):
        self.transforms = transforms
        self.generator = generator

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x, generator=self.generator)
        return x


class RandomFlip:
    def __init__(self, dim=-1, p=0.5):
        self.dim = dim
        self.p = p

    def __call__(self, x, generator=None):
        if generator._rand(size=(1,), low=0, high=1).item() < self.p:
            x = torch.flip(x, dims=(self.dim,))
        return x


class RandomRotate90:
    def __init__(self, dims=(-2, -1), p=0.5):
        self.dims = dims
        self.p = p

    def __call__(self, x, generator=None):
        if generator._rand(size=(1,), low=0, high=1).item() < self.p:
            k = generator._randint(size=(1,), low=1, high=4).item()
            x = torch.rot90(x, k=k, dims=self.dims)
        return x
