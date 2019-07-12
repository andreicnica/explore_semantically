import torch


def horisontal_flip(images, targets):
    """
        Code source https://github.com/eriklindernoren/PyTorch-YOLOv3
    """

    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
