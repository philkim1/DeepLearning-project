#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import json
import logging
import os
import sys

from torchvision import datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            test_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss = test_loss/len(test_loader.dataset)
    total_accuracy = correct/len(test_loader.dataset)
    
    logger.info("Testing loss: {:.4f}".format(total_loss))
    logger.info("Accuracy: {:.4f} ({:.0f}%)".format(total_accuracy,100.0*total_accuracy))
          
        
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    

def train(model, train_loader, criterion, optimizer, epoch, test_loader):

    model.train()
    running_loss = 0
    count = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        count += 1

        logger.info("Train Epoch: {} [{}/{} ({:.0f}%)] Avg Loss: {:.4f}".format(
                    epoch,
                    count*len(data),
                    len(train_loader.dataset),
                    100.0 * count*len(data) / len(train_loader.dataset),
                    running_loss/len(train_loader))
        )
            
    test(model, test_loader, criterion)
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    
def net():
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,256),
                             nn.ReLU(),
                             nn.Linear(256,133))
    
    return model
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

def create_data_loaders(args):
    
    train_dir = args.data_dir + '/train'
    test_dir = args.data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = args.test_batch_size)
    
    return trainloader, testloader
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
    for epoch in range(1, args.epochs + 1):
        
        train_loader, test_loader = create_data_loaders(args)
    
        '''
        TODO: Call the train function to start training your model
        Remember that you will need to set up a way to get training data from S3
        '''
        
        train(model, train_loader, criterion, optimizer, epoch, test_loader)
    
        '''
        TODO: Test the model to see its accuracy
        '''
        #test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "dog_model.pth")
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    args=parser.parse_args()
    
    main(args)
