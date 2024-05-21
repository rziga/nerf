import lightning as L
from torch.optim import Adam
from torch.nn import functional as F
from torchmetrics import PeakSignalNoiseRatio

from .model import MobileR2L


class MobileR2LLighningModule(L.LightningModule):

    def __init__(self, learning_rate, in_channels, emb_dim, hidden_channels, num_layers, num_sr_modules):
        super().__init__()
        self.save_hyperparameters()
        
        # model and loss
        self.model = MobileR2L(
            in_channels, emb_dim, hidden_channels, num_layers, num_sr_modules)
        self.loss_fn = F.mse_loss

        # metrics
        self.train_psnr = PeakSignalNoiseRatio()
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), self.hparams["learning_rate"])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # forward pass
        rays, imgs = batch
        pred_imgs = self.model(rays)
        loss = self.loss_fn(pred_imgs, imgs)

        # log metrics
        self.train_psnr(pred_imgs, imgs)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_psnr", self.train_psnr, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # forward pass
        rays, imgs = batch
        pred_imgs = self.model(rays)
        loss = self.loss_fn(pred_imgs, imgs)

        # log metrics
        self.val_psnr(pred_imgs, imgs)
        self.log("val_psnr", self.val_psnr, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # forward pass
        rays, imgs = batch
        pred_imgs = self.model(rays)
        loss = self.loss_fn(pred_imgs, imgs)

        # log metrics
        self.test_psnr(pred_imgs, imgs)
        self.log("test_psnr", self.test_psnr)
        return loss