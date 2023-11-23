import torch
import torch.nn.functional as F
import torch.nn as nn

class AutoencoderConv(torch.nn.Module):
    def __init__(self):
        super(AutoencoderConv, self).__init__()

        self.encoder = nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            # torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, 3),
            nn.Sigmoid()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
    
class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x