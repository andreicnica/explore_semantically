from train_loop.train_base import TrainBase
import torch.nn.functional as F
import torch
import torchvision
import cv2
import numpy as np


class TrainDefault(TrainBase):
    def __init__(self, cfg, train_loader, test_loader, model, optimizer, device, saver, logger,
                 scheduler=None):
        super().__init__(cfg, train_loader, test_loader, model, optimizer, device, saver, logger,
                         scheduler=scheduler)
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
        all_loss = []

        loss_f = torch.nn.BCEWithLogitsLoss()

        # for self.batch_idx, (imgs, boxes, segmentation) in enumerate(train_loader):
        for self.batch_idx, data in enumerate(train_loader):
            if self.scheduler is not None:
                self.scheduler(optimizer, self.batch_idx, epoch, self.best_train_loss.best[0])

            # Move to device
            for x in data:
                x.detach_()

            imgs = data[:-1]
            segmentation = [data[-1]]

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
                all_loss += log["loss"]

                log = self.get_base_log(reset=True)

        mean_loss = np.mean(all_loss)

        info = {}
        return mean_loss, info

    def _eval(self):
        model = self.model
        device = self.device
        test_loader = self.test_loader
        log_freq = self.batch_log_freq
        epoch = self.epoch
        logger = self.logger
        log = self.get_base_log()
        loss_f = torch.nn.BCEWithLogitsLoss()
        log["epoch"].append(self.epoch)

        with torch.no_grad():
            for self.batch_idx, data in enumerate(test_loader):
                for x in data:
                    x.detach_()

                imgs = data[:-1]
                segmentation = [data[-1]]

                batch_idx = self.batch_idx

                imgs = [x.to(device) for x in imgs]
                segmentation = [x.to(device) for x in segmentation]

                target = segmentation[0]

                predict = model(imgs)

                loss = loss_f(predict, target)

                # -- Update log
                log["loss_eval"].append(loss.item())
                log["batch_idx"].append(self.batch_idx)

                if (batch_idx + 1) % log_freq == 0:
                    logger.write(log)

        logger.write(log)

        mean_loss = np.mean(log['loss_eval'])
        info = {}
        return mean_loss, info

    def _save(self):
        raise NotImplemented

    def _load(self):
        raise NotImplemented
