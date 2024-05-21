import lightning as L

from data.datamodule import BlenderDataModule
from model.lightningmodule import MobileR2LLighningModule


if __name__ == "__main__":
    data_path = "../../../MobileR2L/model/teacher/ngp_pl/Pseudo/lego"
    datamodule = BlenderDataModule(data_path, 1, 16, rand=True)
    model = MobileR2LLighningModule(5e-4, 16*3, 10, 128, 16, 3)
    trainer = L.Trainer(
        callbacks=[]
    )
    
    trainer.fit(
        model,
        datamodule,
    )
    trainer.test(
        model,
        datamodule
    )

    #from model.model import MobileR2L
    #import torch
    #import time
#
    #model = MobileR2L(16*3, 10, 128, 16, 3).cuda()
    #x = torch.randn(1, 16*3, 100, 100).cuda()
    #y = torch.randn(1, 3, 800, 800).cuda()
    #y_pred = model(x)
    #loss = torch.nn.functional.mse_loss(y_pred, y)
    #loss.backward()
#
    #print(y_pred.shape)
#
    #time.sleep(10)