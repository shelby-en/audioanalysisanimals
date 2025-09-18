import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from librosa import amplitude_to_db
import os
import time
import lightning.pytorch as pl 
from torchvision import transforms, models
import torchaudio

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.optim as optim 
from torchmetrics import MetricCollection, Accuracy, F1Score

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import torch

nClasses = 14
batch_size = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nWorkers = 0 if torch.cuda.is_available() else 0
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

specTransforms = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(amplitude_to_db(x))),
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=30),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1,0))], p=0.3)
])

predTransforms = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(amplitude_to_db(x).astype('float32'))),
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
])

class SpectrogramDataset(Dataset):
    def __init__(self, labels, audioDir, transform=specTransforms):
        self.labels = pd.read_csv(labels, header = 0)
        self.audioDir = audioDir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # print(self.labels.iloc[idx])
        # print(self.labels.iloc[idx][2] + '_' + str(self.labels.iloc[idx][1]) + '.npy')
        # sample_path = os.path.join(self.audioDir, self.labels.iloc[idx][2] + '_' + str(self.labels.iloc[idx][0]) + '.npy')
        sample_path = os.path.join(self.audioDir, str(self.labels.iloc[idx][0]) + '.npy')
        s = torch.from_numpy(np.load(sample_path).astype('float32'))
        label = self.labels.iloc[idx][1:]
        # s = (s - torch.mean(s)) / (torch.std(s) + 1e-6)  # standardization
        if self.transform:
            s = self.transform(s)

        return s, torch.tensor(label, dtype=torch.float32)

class SepConv(nn.Module):
    """Depthwise-separable conv: very lightweight."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvTest(pl.LightningModule):
    def __init__(self, num_classes = nClasses, learning_rate = 0.001, ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.loss_fun = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes


        # Feature extractor (very small):
        # Input: [B, 1, H, W]
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Downsample a bit with stride=2 to reduce compute; keep channels modest
        self.block1 = SepConv(16, 32, stride=2)   # [B, 32, H/2,  W/2]
        self.block2 = SepConv(32, 64, stride=2)   # [B, 64, H/4,  W/4]
        self.block3 = SepConv(64, 96, stride=2)   # [B, 96, H/8,  W/8]

        # Optional regularization
        self.dropout = nn.Dropout(p=0.2)

        # Global average pooling -> [B, C, 1, 1] -> [B, C]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Tiny classifier head
        self.classifier = nn.Linear(96, num_classes)

        self.val_acc = F1Score(task='multilabel', num_classes=nClasses, num_labels=nClasses, threshold=0.2)
        self.train_acc = F1Score(task='multilabel', num_classes=nClasses, num_labels=nClasses, threshold=0.2)
        self.test_acc = F1Score(task='multilabel', num_classes=nClasses, num_labels=nClasses, threshold=0.2)

    def forward(self, x):  # x: [B, 1, H, W]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.gap(x)                # [B, 96, 1, 1]
        x = x.view(x.size(0), -1)      # [B, 96]
        logits = self.classifier(x)    # [B, num_classes]
        return logits
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        # print(x.shape)
        # print(y)

        # y = torch.flatten(y, start_dim=0)
        output = self(x)
        # print(output)

        loss = self.loss_fun(output, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_acc(output, y)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
    
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        
        # y = torch.flatten(y,start_dim=0) ## Ensure you reshape your landmarks data to compare against the preds
        output = self(x)
        # print(y)

        loss = self.loss_fun(output, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_acc(output, y)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)

        self.log("timestamp", float(time.time()), on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x,y = batch

        # y = torch.flatten(y,start_dim=0) ## Ensure you reshape your landmarks data to compare against the preds
        output = self(x)

        loss = self.loss_fun(output, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.test_acc(output, y)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def predict_step(self, x):
        output = self(x)

        return output, x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = 0.005)
        return optimizer
    
    def train_dataloader(self):
        return train_loader
    
    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

def load_data(labelPath, dataPath):
    global train_loader, val_loader, test_loader
    ds = SpectrogramDataset(labelPath, dataPath)

    train_loader = DataLoader(SpectrogramDataset(labelPath, dataPath), batch_size=batch_size, shuffle=True, num_workers=nWorkers)
    val_loader = DataLoader(SpectrogramDataset(labelPath, dataPath, transform=predTransforms), batch_size=batch_size, shuffle=False, num_workers=nWorkers)
    test_loader = DataLoader(SpectrogramDataset(labelPath, dataPath, transform=predTransforms), batch_size=batch_size, shuffle=False, num_workers=nWorkers)

    # train, validate, test = random_split(ds, [0.7, 0.1, 0.2])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=nWorkers)
    # val_loader = DataLoader(validate, batch_size=batch_size, shuffle=False, num_workers=nWorkers)
    # test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=nWorkers)

    return ds, train_loader, val_loader, test_loader

def config_model(nn, log_dir, chk_dir, max_epochs = 100, nClasses = nClasses):
    model = nn(nClasses)

    progress_bar_task = RichProgressBar(refresh_rate=1, leave=False,
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82"
        )
    )

    version_num = len(os.listdir(f"{log_dir}/lightning_logs/"))

    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=chk_dir,
            save_top_k=1,        # save the best model
            mode="min",
            every_n_epochs=1,
            filename=f'version_{version_num}'
        )

    early_stopping = EarlyStopping('val_loss', patience = 3, mode = 'min')

    # Train and test the model
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1, #if torch.cuda.is_available() else None,  
        max_epochs=max_epochs,
        check_val_every_n_epoch=2,
        callbacks=[progress_bar_task, checkpoint_callback, early_stopping],
        logger=CSVLogger(save_dir=log_dir),
        # log_every_n_steps=1,
    )

    return model, trainer, version_num


def main():
    ds, train_loader, val_loader, test_loader = load_data('./data/multi_labels_ol.csv', './data/spectrograms_ol')
    print(ds[0][1])
    model, trainer, version_num = config_model(ConvTest, './data/chkpts/lightning/logs/', './data/chkpts/lightning/chks/',max_epochs=100)

    # import librosa
    # i = 0
    # for data in ds:
    #     y = data[1]
    #     # print(y)
    #     data = data[0].numpy().squeeze()
    #     assert np.isnan(data).any() == False
    #     print(np.max(data), np.min(data), np.sum(data))
    #     # plt.figure(i)
    #     # librosa.display.specshow(data)
    #     # plt.savefig(f"spec{i}")
    #     # i += 1

    # exit()

    model.train()
    trainer.fit(model, train_loader, val_loader)
    model.eval()
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()