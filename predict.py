import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(x, x)
        self.fc2 = nn.Linear(x, x)
        self.fc3 = nn.Linear(x, x)
        self.fc4 = nn.Linear(x, x)
        
        self.dropout = nn.Dropout(p = .2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x


ap = argparse.ArgumentParser()
ap.add_argument("path_to_image", type=str, help="path to input image")
ap.add_argument("checkpoint", type=str, help="checkpoint file containing model")
ap.add_argument("--top_k", "-tk", default=1, required=False, type=int, help="type of model")
ap.add_argument("--category_names", "-lr", type=str, default="cat_to_name.json", required=False, help="mapping of categories to convert to real names of the flowers")
ap.add_argument("--gpu", "-g", action="store_true", help="If called model will run on gpu")


args = vars(ap.parse_args())


if args["gpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(args["category_names"], 'r') as f:
    cat_to_name = json.load(f)
    
    
def Load_Model(FilePath):
    checkpoint = torch.load(FilePath)
    model = models.densenet121(pretrained = True)
    
    model.classifier = checkpoint['Classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    class_to_idx = checkpoint['class_to_idx']
    
    optimizer = optim.Adam(model.classifier.parameters(), lr =.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, epoch, class_to_idx, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    picture = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    pic2 = transform(picture)
    
    np_image = np.array(pic2)
    
    return np_image

def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
          
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std *  image + mean

    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    rebuilt_model, epochs, class_to_idx, optimizer = Load_Model(model)
    input_image = torch.tensor(process_image(image_path))
    input_image = torch.unsqueeze(input_image,0)
    input_image = input_image.to(device)
    rebuilt_model.to(device)
    
    rebuilt_model.eval()
    with torch.no_grad():
        output = rebuilt_model.forward(input_image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(args["top_k"], dim=1)
        
    inv_class_to_idx = {v: k for k, v in class_to_idx.items()}
    flatten_top_p = sum(top_p.tolist(), [])
    
    class_id = []
    for x in top_class.tolist()[0]:
        class_id.append(inv_class_to_idx[x])
    
    return flatten_top_p, class_id

probs, classes = predict(args["path_to_image"], args["checkpoint"])
index = probs.index(max(probs))

for x in range(len(classes)):
    classes[x] = cat_to_name[classes[x]]

for x in range(len(probs)):
    print(str(x+1) + ". flower type: " + classes[x] + "\n" + "   probability: " + str(round(probs[x] * 100, 2)))


