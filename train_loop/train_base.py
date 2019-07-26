import numpy as np
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from collections import OrderedDict, Collection
from copy import deepcopy
import time
import torch

from utils.logger import MultiLogger
from utils.save_training import SaveData

"""        
:param header: {key: [synthesis_type, print_name, print_format]
synthesis_type: 
    μ : mean of values
    + : sum
    . : last element
    - : single value
    μσmM : mean, std, min, max
"""

BASE_LOG_HEADER = OrderedDict({
    "epoch": (".", "E", "{}"),
    "batch_idx": (".", "B", "{:06}"),
    "bps": (".", "BPS", "{:04.4f}"),
    "gradientμ": ("μ", "∇μ", "{:.2f}"),
    "gradientstd": ("μ", "∇std", "{:.2f}"),
    "gradientmax": ("μ", "∇max", "{:.2f}"),
    "loss": ("μσmM", "L", "{:.2f}"),
    "loss_eval": ("μσmM", "LE", "{:.2f}"),
})


class BestStats:
    def __init__(self):
        self._best_score = np.inf
        self._iter_no = -1

    def __call__(self, score, iter_no):
        if score < self._best_score:
            self._best_score = score
            self._iter_no = iter_no

    @property
    def best(self):
        return self._best_score, self._iter_no


class TrainBase:
    def __init__(self, cfg: Namespace, train_loader: DataLoader, test_loader: DataLoader,
                 model: Module, optimizer: Optimizer, device: str,
                 saver: SaveData, logger: MultiLogger,
                 scheduler=None):

        self.cfg = cfg
        self.out_dir = getattr(cfg, "out_dir")
        self.batch_log_freq = getattr(cfg, "batch_log_freq")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.saver = saver
        self.logger = logger
        self.scheduler = scheduler

        self.best_train_loss = BestStats()
        self.best_eval_loss = BestStats()

        self.epoch = 0
        self.batch_idx = 0
        self.loss = np.inf

        self.header = BASE_LOG_HEADER
        self.base_log = OrderedDict()
        self.start_batch_idx = 0
        self.start_time = 0
        self._log = None

    def train(self):
        print(f"Train {self.epoch}")

        if self.epoch == 0:
            self.first_train()

        self.batch_idx = 0

        self.model.train()

        self.loss, info = self._train()
        self.best_train_loss(self.loss, self.epoch)
        self.logger.info(f"Best train loss: {self.best_train_loss.best[0]} @ epoch "
                         f"{self.best_train_loss.best[1]}")

        self.epoch += 1

        return self.loss, info

    def first_train(self):
        self.logger.set_header(self.header)
        self.base_log = OrderedDict({k: [] for k in self.header.keys()})
        self._log = self.get_base_log(reset=True)

    def get_base_log(self, reset=False):
        self.start_batch_idx = self.batch_idx
        self.start_time = time.time()
        self._log = deepcopy(self.base_log)

        return self._log

    def std_update_log(self, log: OrderedDict):
        log["epoch"].append(self.epoch)
        log["batch_idx"].append(self.batch_idx)

        bps = float(self.batch_idx - self.start_batch_idx + 1) / (time.time() - self.start_time)
        log["bps"].append(bps)

        g_mean, g_std, g_max = self.get_gradient_info(self.model)

        log["gradientμ"].append(g_mean)
        log["gradientstd"].append(g_std)
        log["gradientmax"].append(g_max)

    def eval(self):
        print(f"Eval")

        if len(self.base_log) <= 0:
            self.first_train()

        self.model.eval()
        mean_loss, info = self._eval()
        self.best_eval_loss(mean_loss, self.epoch)

        save_data = self.save()
        self.saver.save_training_data(save_data, mean_loss)

        self.logger.info(f"Best EVAL loss: {self.best_eval_loss.best[0]} @ epoch "
                         f"{self.best_eval_loss.best[1]}")

        return mean_loss, info

    def save(self):
        save_data = {
            'epoch': self.epoch,
            'batch_idx': self.batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }

        return save_data

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

        self._load(checkpoint)

        return True

    def get_gradient_info(self, model):
        grad = [p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None]
        if len(grad) == 0:
            return 0,0,0
        return np.mean(grad), np.std(grad), np.max(grad)

    def _train(self):
        raise NotImplemented

    def _eval(self):
        raise NotImplemented

    def _save(self):
        pass

    def _load(self, checkpoint):
        pass
