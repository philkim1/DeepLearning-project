import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
import io
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def net():
    
    model = models.resnet50(pretrained=True)
    
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,256),
                             nn.ReLU(),
                             nn.Linear(256, 133))

    return model
    
    
JPEG_CONTENT_TYPE = 'image/jpeg'


def model_fn(model_dir):
    
    model = net() 
     
    with open(os.path.join(model_dir, "finetune_dog_model.pth"), "rb") as f: 
        model.load_state_dict(torch.load(f))
    
    model.eval()

    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    
    if content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body))
        
        return image
        
    else:
        print('ping....')
        logger.critical(f'unsupported content type: {content_type}')
    

def predict_fn(input_object, model):
    
    transform_test_valid = transforms.Compose([transforms.Resize((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    
    input_object=transform_test_valid(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))


    return prediction