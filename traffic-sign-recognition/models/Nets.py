from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    '''
    Build LeNet Model
    '''
    def __init__(self, args):
        super(LeNet, self).__init__() 

 
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(args.num_channels, 32, 5, 1, 2),   # padding=2
            nn.ReLU(),                  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)
        )

     
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),  # input_size=(6*14*14)  
            nn.ReLU(),            # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)    # output_size=(16*5*5)
        )

 
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  
            nn.ReLU()                    
        )

        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

 
        self.fc3 = nn.Linear(84, 62)

  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
 
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
