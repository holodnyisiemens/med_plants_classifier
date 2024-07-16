import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision.models import DenseNet121_Weights

class DenseNetClassifier(pl.LightningModule):
    def __init__(self, in_channels, out_classes, eta):
        super().__init__()
        self.save_hyperparameters()

        # pretrained
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.criterion = nn.CrossEntropyLoss()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metrics = {'accuracy': Accuracy(task='multiclass', num_classes=out_classes).to(device)}

        self.preds_stage = {"train": {"loss": [], "accuracy":[]},
                        "valid": {"loss": [], "accuracy":[]},
                        "test": {"loss": [], "accuracy":[]}}

    def forward(self, x):
        return self.model(x)

    # operations with each batch
    def shared_step(self,
                    sample,
                    stage):

        x, y = sample
        logits = self.forward(x['image'].to(torch.float32))

        # get index of maximum value (0 axis - batch, 1 - channel)
        preds = torch.argmax(logits, 1)

        loss = self.criterion(logits, y.to(torch.int64))

        self.preds_stage[stage]['loss'].append(loss.detach().cpu())
        self.preds_stage[stage]['accuracy'].append(self.metrics["accuracy"](preds, y).detach().cpu())
        return loss

    # common operations for each stage
    def shared_epoch_end(self, stage):
        loss = self.preds_stage[stage]['loss']
        loss = torch.stack(loss)
        loss = np.mean([x.item() for x in loss])

        acc = self.preds_stage[stage]['accuracy']
        acc = torch.stack(acc)
        acc = np.mean([x.item() for x in acc])

        # for logs
        metrics = {
            f'{stage}_loss': loss,
            f'{stage}_acc': acc,
        }

        self.log_dict(metrics, prog_bar=True)

        self.preds_stage[stage]['loss'].clear()
        self.preds_stage[stage]['accuracy'].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.eta)

        scheduler_dict = {
            # if the specified metric does not change, the learning rate reduces
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                # how many inefficient epochs to wait before reducing the learning rate
                patience=1
            ),
            'interval': 'epoch',
            'monitor': 'valid_loss'
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    # steps (use DataLoader to load data for corresponding stage)
    def training_step(self, batch, batch_idx):
        return self.shared_step(sample=batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(sample=batch, stage="valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(sample=batch, stage="test")

    # metrics calculation
    def on_training_epoch_end(self):
        return self.shared_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(stage="valid")

    def on_test_epoch_end(self):
        return self.shared_epoch_end(stage="test")
