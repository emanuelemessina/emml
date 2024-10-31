import math


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params


def get_mean_stdev(dataset):
    mean = dataset.data.float().mean() / 255
    std = dataset.data.float().std() / 255

    print(f"mean: {mean}")
    print(f"stdev: {std}")

    return mean, std


def num_batches(loader):
    return math.ceil((len(loader.dataset) / loader.batch_size))
