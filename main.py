import logging
import os
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from clearml import Task
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import ArgsType, LightningCLI, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection
from torchvision import datasets, transforms

import optimizers
from models import get_model
from print_logo import print_logo

logger = logging.getLogger()


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root: str = "./data",
                 image_size: int = 224,
                 num_workers: int = 4,
                 train_batch_size: int = 128,
                 val_batch_size: int = 128,
                 use_tencrop_test: bool = True):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.use_tencrop_test = use_tencrop_test

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self._setup_dataset()

    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda image: transforms.Resize(random.randint(256, 480))(image)), # resize shortest
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    @property
    def val_transforms(self):
        if self.use_tencrop_test:
            val_transforms = transforms.Compose([
                transforms.Lambda(lambda image: [transforms.TenCrop(size=(self.image_size, self.image_size))(img) \
                                                    for img in [transforms.Resize(224)(image),
                                                                transforms.Resize(256)(image),
                                                                transforms.Resize(384)(image),
                                                                transforms.Resize(480)(image),
                                                                transforms.Resize(640)(image)]]),
                transforms.Lambda(lambda tuples: [img for tup in tuples for img in tup]),
                # transforms.Resize(256),
                # transforms.TenCrop(size=(self.image_size, self.image_size)),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda tensors: torch.stack([transforms.Normalize(mean=self.mean, std=self.std)(t) for t in tensors])),
            ])
        else:
            val_transforms = transforms.Compose([
                transforms.Resize(size=(self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        return val_transforms

    def _setup_dataset(self):
        self.train_data = datasets.ImageFolder(root=os.path.join(self.data_root, "train"), transform=self.train_transforms)
        self.val_data = datasets.ImageFolder(root=os.path.join(self.data_root, "val"), transform=self.val_transforms)
        logger.info(f"Setup ImageNet dataset successfully with {len(self.train_data)} train and {len(self.val_data)} val samples!")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)


class ClassificationModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        self._build_loss()
        self._build_model()
        self._build_metrics()

    def _build_loss(self):
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def _build_metrics(self):
        metrics = MetricCollection({"top1_accuracy": Accuracy(task="multiclass", top_k=1,
                                                                num_classes=self.num_classes),
                                    "top5_accuracy": Accuracy(task="multiclass", top_k=5,
                                                                num_classes=self.num_classes)})
        self.train_metrics = metrics.clone(postfix="/train")
        self.val_metrics = metrics.clone(postfix="/val")

    def _build_model(self):
        self.model = get_model(self.model_name, self.num_classes, self.pretrained)

    def loss(self, preds, targets):
        return self.loss_fn(preds, targets)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}

    def training_step_end(self, outputs):
        self.trainer.fit_loop.running_loss.append(outputs["loss"])
        self.log("loss/train", outputs["loss"].item(), logger=True, sync_dist=True)
        self.train_metrics.update(outputs["y_hat"], outputs["y"])
        return {'loss': outputs["loss"]}
    
    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss_epoch/train", loss, logger=True, sync_dist=True)
        calculated_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        calculated_metrics = {f"{key}": value for key, value in calculated_metrics.items()}
        self.log_dict(calculated_metrics, logger=True, sync_dist=True)
        super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(x.shape) == 5:
            B, NC, C, H, W = x.shape  # batch, num_crop, channel, height, width
            x = x.view(-1, C, H, W)
            y_hat = self.forward(x)
            y_hat = y_hat.view(B, NC, -1)
            y_hat = torch.sum(y_hat, dim=1) / NC
        else:
            y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss, 'y_hat': y_hat, 'y': y}
    
    def validation_step_end(self, outputs):
        # self.log("loss_val", outputs["loss"].item(), logger=True)
        self.val_metrics.update(outputs["y_hat"], outputs["y"])
        return {'loss': outputs["loss"]}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss_epoch/val", loss, logger=True, sync_dist=True)
        calculated_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        calculated_metrics = {f"{key}": value for key, value in calculated_metrics.items()}
        self.log_dict(calculated_metrics, logger=True, sync_dist=True)
        super().validation_epoch_end(outputs)


class ImageNetTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        self._add_optimizer_and_lrscheduler_args(parser)
        parser.add_argument("--use_clearml", type=bool,
                            help="Whether to use ClearML or not")
        parser.add_argument("--clearml_project_name", type=str,
                            help="ClearML Project name")
        parser.add_argument("--clearml_task_name", type=str,
                            help="ClearML Task name")
        self._add_callbacks(parser)

    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler):
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }

    def _add_optimizer_and_lrscheduler_args(self, parser):
        parser.add_optimizer_args((
            torch.optim.SGD,
            torch.optim.Adam
        ))
        parser.add_lr_scheduler_args((
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            torch.optim.lr_scheduler.MultiStepLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
            optimizers.LinearWarmupCosineAnnealingLR,
            ReduceLROnPlateau
        ))

    def _add_callbacks(self, parser):
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "step"})
        parser.add_lightning_class_args(ModelCheckpoint, "save_checkpoint")
        parser.set_defaults({
            "save_checkpoint.monitor": "loss_epoch/val",
            "save_checkpoint.dirpath": None,
            "save_checkpoint.filename": "{epoch}-{step}",
            "save_checkpoint.save_top_k": -1,
            "save_checkpoint.save_on_train_epoch_end": True
        })

    def before_instantiate_classes(self):
        if self.config[self.config["subcommand"]]["use_clearml"]:
            task = Task.init(project_name=self.config[self.config["subcommand"]]["clearml_project_name"],
                            task_name=self.config[self.config["subcommand"]]["clearml_task_name"])
            self.parser.parse_args()
        return super().before_instantiate_classes()


def cli_main(args: ArgsType = None):
    print_logo()
    ImageNetTrainingCLI(ClassificationModel, ImageNetDataModule, args=args)

if __name__ == "__main__":
    cli_main()