import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
from torch.nn import functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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
        self.test_ssim = StructuralSimilarityIndexMeasure()
        self.test_lpips = LearnedPerceptualImagePatchSimilarity()
    
    def configure_optimizers(self):
        # optimizer
        optimizer = Adam(self.model.parameters(), self.hparams["learning_rate"])

        #scheduler - linear rampup -> exponential decay
        switch_step = 200
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, 0.1, 1.0, switch_step),
            ExponentialLR(optimizer, 0.99998)
        ], milestones=[switch_step])
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler_config]
    
    def training_step(self, batch, batch_idx):
        # forward pass
        rays, imgs = batch
        pred_imgs = self.model(rays)
        loss = self.loss_fn(pred_imgs, imgs)

        # log metrics
        self.train_psnr(pred_imgs, imgs)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_psnr", self.train_psnr, prog_bar=True)

        # log images
        if (self.trainer.global_step) % 5000 == 0:
            self.logger.log_image(
                key="samples", images=[pred_imgs[0].cpu(), imgs[0].cpu()], caption=["pred", "gt"])

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
        self.test_ssim(pred_imgs, imgs)
        self.log("test_ssim", self.test_lpips)
        self.test_lpips(pred_imgs, imgs)
        self.log("test_lpips", self.test_lpips)
        return loss
