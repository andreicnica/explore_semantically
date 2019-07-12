import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import transforms
import os
from argparse import Namespace

from models import get_model
from train_loop import get_train
from utils.save_training import SaveData
from utils.logger import MultiLogger
from utils.coco_dataset import COCODetection, detection_collate
from utils.ssd_augmentation import SSDAugmentation
from utils import config

def train(epoch, train_loader, model, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data, target)

        loss = model.calculate_loss((data, target), None, output)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(test_loader, model, device):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


def run(cfg: Namespace) -> None:
    use_cuda = cfg.use_cuda
    no_epochs = cfg.no_epochs
    out_dir = cfg.out_dir
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # torch.autograd.set_detect_anomaly(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    config.add_to_cfg(cfg, subgroups=["model", "train"], new_arg='out_dir', new_arg_value=out_dir)

    # ==============================================================================================
    # -- Data loading
    dataset_root = cfg.dataset_root
    data_mean = cfg.data_mean
    no_classes = cfg.no_classes
    max_expand = cfg.max_expand
    in_sizes = cfg.in_sizes

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    # transform = transforms.Compose([
    #         transforms.Normalize(torch.tensor(cfg.norm_mean),
    #                              torch.tensor(cfg.norm_std))
    #     ])
    #
    dataset = COCODetection(root=dataset_root,
                            transform=SSDAugmentation(in_sizes, data_mean,
                                                      no_classes=no_classes, max_expand=max_expand))

    train_loader = data.DataLoader(dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # ==============================================================================================
    # Load model, optimizer and loss

    model = get_model(cfg.model, no_classes=no_classes).to(device)

    _optimizer = getattr(torch.optim, cfg.train.algorithm)
    optim_args = vars(cfg.train.algorithm_args)
    optimizer = _optimizer(model.parameters(), **optim_args)

    # ==============================================================================================
    # Loaders and stuff
    saver = SaveData(cfg.out_dir, save_best=cfg.save_best, save_all=cfg.save_all)
    logger = MultiLogger(cfg.out_dir, cfg.tb, cfg.log_key)
    train_loop = get_train(cfg.train, train_loader, None, model, optimizer, device, saver, logger)

    # ==============================================================================================
    # Train loop

    for epoch in range(no_epochs):
        train_loop.train()
        # test(test_loader, model, device)


def main():
    # Reading args
    args = read_config()  # type: Args
    args.out_dir = "results"

    if not hasattr(args, "out_dir"):
        from time import time
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        out_dir = f'./results/{str(int(time())):s}_{args.experiment:s}'
        os.mkdir(out_dir)
        args.out_dir = out_dir
    else:
        assert os.path.isdir(args.out_dir), "Given directory does not exist"

    if not hasattr(args, "run_id"):
        args.run_id = 0

    run(args)


if __name__ == "__main__":
    from liftoff import parse_opts
    run(parse_opts())
