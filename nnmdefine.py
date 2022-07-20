import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Net,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.layer(x)
        return x
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            Net(input_dim, hidden_dim),
            *[Net(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x