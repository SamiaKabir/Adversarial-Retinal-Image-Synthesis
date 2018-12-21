import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def isBinary(t):
    return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum() == 0

##create  autoencoder for vessel network
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,4,3,2,1),
            nn.ReLU(),
            nn.Conv2d(4,8,3,2,1),
            nn.ReLU(),
            nn.Conv2d(8,16,3,2,1),
            nn.ReLU(),
        )  

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,4,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(4,3,4,2,1),
            # nn.Linear(38*38,32*32),
        )


    def forward(self, x):
        encoded = self.encoder(x)
 
        decoded = self.decoder(encoded)
        # print(decoded.size())
        return decoded

##create generator

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,4,3,2,1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(4,8,3,2,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
        )

        self.encoder3= nn.Sequential(
            nn.Conv2d(8,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )

        self.encoder4= nn.Sequential(

            nn.Conv2d(16,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )
         
        self.encoder5= nn.Sequential(
            nn.Conv2d(16,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )
        self.encoder6= nn.Sequential(
            nn.Conv2d(16,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )  

        self.encoder7= nn.Sequential(
            nn.Conv2d(16,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )  

        self.encoder8= nn.Sequential(
            nn.Conv2d(16,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )  

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder2 = nn.Sequential(                
            nn.ConvTranspose2d(16,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2),
        )

        self.decoder3 = nn.Sequential(   
            nn.ConvTranspose2d(16,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2),
        )

        self.decoder4 = nn.Sequential(   
            nn.ConvTranspose2d(16,16,4,2,1),
            nn.BatchNorm2d(16),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2),
        )

        self.decoder5 = nn.Sequential(   
            nn.ConvTranspose2d(16,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )

        self.decoder6 = nn.Sequential(   
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
        )

        self.decoder7 = nn.Sequential(   
            nn.ConvTranspose2d(8,4,4,2,1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
        )

        self.decoder8 = nn.Sequential(  
            nn.ConvTranspose2d(4,3,4,2,1),
            # nn.Tanh(),
        )

        self.activate_s= nn.Sequential(  
            nn.Sigmoid(),
        )
        self.activate_t= nn.Sequential(  
            nn.Tanh(),
        )

    def forward(self, x):
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.encoder3(encoded2)
        encoded4 = self.encoder4(encoded3)
        encoded5 = self.encoder5(encoded4)
        encoded6 = self.encoder6(encoded5)
        encoded7 = self.encoder7(encoded6)
        encoded8 = self.encoder8(encoded7)

        # print(encoded.size())
        # x= encoded.view(-1)
        decoded1 = self.decoder1(encoded8)
        decoded1= decoded1+ encoded7
        decoded2 = self.decoder2(decoded1)
        decoded2= decoded2 + encoded6
        decoded3 = self.decoder3(decoded2)
        decoded3= decoded3+ encoded5
        decoded4 = self.decoder4(decoded3)
        decoded4= decoded4+ encoded4
        decoded5 = self.decoder5(decoded4)
        decoded5= decoded5+ encoded3
        decoded6 = self.decoder6(decoded5)
        decoded6= decoded6+ encoded2
        decoded7 = self.decoder7(decoded6)
        decoded7= decoded7+ encoded1
        decoded8 = self.decoder8(decoded7)

        if isBinary(decoded8) :
            decoded8=self.activate_s(decoded8)
        else:
            decoded8=self.activate_t(decoded8)
        # print(decoded.size())
        return decoded8


##Create discremenator 
class discremenator(nn.Module):
    def __init__(self):
        super(discremenator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,4,3,2,1),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(4,8,3,2,1),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.encoder3= nn.Sequential(
            nn.Conv2d(8,16,3,2,1),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.encoder4= nn.Sequential(

            nn.Conv2d(16,16,3,2,1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.encoder3(encoded2)
        encoded4 = self.encoder4(encoded3)
        return encoded4


### Create instance of the networks
Net= net()
Generator = generator()
Discremenator= discremenator()

## Loading the training and test sets

# Converting the images for tifImage to tensor, so they can be accepted as the input to the network
transform = transforms.Compose(
    [
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ]
)


transform2 = transforms.Compose(
    [
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ]
)

trainset = torchvision.datasets.ImageFolder(root='./dataset/train', transform= transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

Gtruth = torchvision.datasets.ImageFolder(root='./dataset/GroundTruth', transform= transform2, target_transform=None)
Gtruthloader = torch.utils.data.DataLoader(Gtruth, batch_size=1, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='./dataset/test_vessel', transform=transform, target_transform=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

Gtest = torchvision.datasets.ImageFolder(root='./dataset/test_fundus', transform=transform2, target_transform=None)
Gtestloader = torch.utils.data.DataLoader(Gtest, batch_size=1, shuffle=False)

vesselset = torchvision.datasets.ImageFolder(root='./dataset/vessel_net', transform=transform, target_transform=None)
vesselloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


### Define the loss and create your optimizer
optimizer_D = torch.optim.Adam(Discremenator.parameters(), lr=0.001)
optimizer_G= torch.optim.Adam(Generator.parameters(), lr=0.001)
optimizer= torch.optim.Adam(Net.parameters(), lr=0.001)
loss_func = nn.MSELoss()
loss_func_d=nn.BCELoss()

## vessel autoencoder training

for epoch in range(20):
    
    for i, data in enumerate(vesselloader, 0):
        
		## Getting the input and the target from the training set
        input, dummy = data
        # input= torch.nn.functional.interpolate(input,size=512,)

        target = input
        pred= Net(input)

        loss = loss_func(pred, target)      # mean square error
        if i%10000==0 :
           print(loss.item())
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients


### Main training loop
for epoch in range(30):
    
    for i, (data1,data2) in enumerate(zip(trainloader,Gtruthloader)):
        # print(input)

        input,dummy=data1
        target,label=data2
		## Getting the input and the target from the training set
        # input, target = data
       

        real=Discremenator(input)
        temp= Net(input)
        temp2= Generator(temp)
        fake= Discremenator(temp2)
        # input= torch.nn.functional.interpolate(input,size=512,)
        
        # target = input
        # pred= Generator(input)

        # target=target.type(torch.FloatTensor)
        input= fake
        target= real
        loss_d = loss_func_d(fake, real.detach())      # entropy loss
        # if i%10000==0 :
        #    print(loss.item())


        optimizer_D.zero_grad()               # clear gradients for this training step
        loss_d.backward()                     # backpropagation, compute gradients
        optimizer_D.step()                    # apply gradients
        
    for j, (data3,data4) in enumerate(zip(trainloader,Gtruthloader)):
        # print(input)

        input,dummy=data3
        target,label=data4
		## Getting the input and the target from the training set
        # input, target = data 
        # input=Net(input)
        # target = input
        temp=Net(input)
        pred= Generator(temp)

        # target=target.type(torch.FloatTensor)

        loss = loss_func(pred, target)      # mean square error
        if j%10000==0 :
           print(loss.item())

        optimizer_G.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer_G.step()                   # apply gradients


### Testing the network on 10,000 test images and computing the loss
loss=0
with torch.no_grad():
    for i, (data1,data2) in enumerate(zip(trainloader,Gtruthloader)):
        # print(input)
        
        input,dummy=data1
        target,label=data2
        # input, dummy = data
        # # input= torch.nn.functional.interpolate(input,size=512,)
        # target = input

        temp=Net(input)
        output= Generator(temp) 
        loss = loss_func(output, target)      # mean square error
        
print("Loss in Test Results....")
print(loss)

### Displaying or saving the results as well as the ground truth images for the first five images in the test set

#show image
def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(img)

for i, (data1,data2) in enumerate(zip(testloader,Gtestloader)):
        # print(input)

        input,dummy=data1
        target,label=data2
        # input, dummy = data
        # target = input
        temp=Net(input)
        output= Generator(temp) 
       # get some random training images
        images  = output

       # show images
        torchvision.utils.save_image(images,'out.png')
        imshow(torchvision.utils.make_grid(images))
        # imshow(images)
        # plt.savefig(images)
    
        # img = transforms.ToPILImage()(images)
        # img.save('output.png')
        break
        # print labels
        print(' '.join())

