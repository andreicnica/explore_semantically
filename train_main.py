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
from utils.lr_scheduler import LR_Scheduler


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
    # torch.multiprocessing.set_start_method("spawn")

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
    data_std = cfg.data_std
    no_classes = cfg.no_classes
    max_expand = cfg.max_expand
    in_sizes = cfg.in_sizes

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    train_dataset = COCODetection( root=dataset_root, image_set=cfg.train_image_set,
                                   transform=SSDAugmentation(in_sizes, data_mean, data_std, no_classes=no_classes,
                                                             max_expand=max_expand))

    train_loader = data.DataLoader(train_dataset, batch_size,
                                   num_workers=num_workers,
                                   shuffle=True,  # collate_fn=detection_collate,
                                   pin_memory=True)

    val_dataset = COCODetection(root=dataset_root, image_set=cfg.val_image_set,
                                transform=SSDAugmentation(in_sizes, data_mean, data_std, no_classes=no_classes,
                                                          max_expand=max_expand, validation=True))

    val_loader = data.DataLoader(val_dataset, batch_size,
                                 num_workers=num_workers,
                                 shuffle=True,  # collate_fn=detection_collate,
                                 pin_memory=True)

    # ==============================================================================================
    # Loaders and stuff
    saver = SaveData(cfg.out_dir, save_best=cfg.save_best, save_all=cfg.save_all)
    logger = MultiLogger(cfg.out_dir, cfg.tb, cfg.log_key)

    # ==============================================================================================
    # Load model, optimizer and loss

    model = get_model(cfg.model, no_classes=no_classes)

    if torch.cuda.device_count() > 1 and len(cfg.gpus) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=cfg.gpus)

    model = model.to(device)

    _optimizer = getattr(torch.optim, cfg.train.algorithm)
    optim_args = vars(cfg.train.algorithm_args)
    optimizer = _optimizer(model.parameters(), **optim_args)

    lr_scheduler = cfg.train.lr_scheduler
    scheduler = None

    if lr_scheduler.use:
        scheduler = LR_Scheduler(lr_scheduler.mode, lr_scheduler.lr, lr_scheduler.epochs,
                                 iters_per_epoch=len(train_loader),
                                 lr_step=getattr(lr_scheduler, "lr_step", 0),
                                 warmup_epochs=getattr(lr_scheduler, "warmup_epochs", 0), logger=logger,)

    # ==============================================================================================
    # Training loop

    train_loop = get_train(cfg.train, train_loader, val_loader, model, optimizer, device,
                           saver, logger, scheduler=scheduler)

    # ==============================================================================================
    # Train loop

    for epoch in range(no_epochs):
        train_loop.train()
        train_loop.eval()


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
