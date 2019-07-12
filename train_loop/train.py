from train_loop.train_base import TrainBase
import torch.nn.functional as F
import torch
import torchvision
import cv2
import numpy as np


class TrainDefault(TrainBase):
    def __init__(self, cfg, train_loader, test_loader, model, optimizer, device, saver, logger):
        super().__init__(cfg, train_loader, test_loader, model, optimizer, device, saver, logger)
        self.view_freq = 0

    def _train(self):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        train_loader = self.train_loader
        log_freq = self.batch_log_freq
        epoch = self.epoch
        logger = self.logger
        log = self.get_base_log()
        loss_f = torch.nn.BCEWithLogitsLoss()

        for self.batch_idx, (imgs, boxes, segmentation) in enumerate(train_loader):
            batch_idx = self.batch_idx

            # Move to device
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            segmentation = [x.to(device, non_blocking=True) for x in segmentation]
            target = segmentation[0]

            optimizer.zero_grad()

            predict = model(imgs)

            loss = loss_f(predict, target)

            loss.backward()
            optimizer.step()

            # -- Update log
            # TODO caution, might be slow to do update each step
            self.std_update_log(log)  # Standard update log
            log["loss"].append(loss.item())

            if (batch_idx + 1) % log_freq == 0:

                logger.write(log)

                log = self.get_base_log(reset=True)

        mean_loss = 0
        info = {}
        return mean_loss, info

    def _eval(self):
        pass

    def _save(self):
        raise NotImplemented

    def _load(self):
        raise NotImplemented
