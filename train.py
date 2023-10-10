import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=str, help="directory containing data")
ap.add_argument("--arch", "-a", required=False, help="type of model")
ap.add_argument("--learning_rate", "-lr", type=float, default=.003, required=False, help="learning rate of the model")
ap.add_argument("--hidden_units", "-hu", type=int, default=512, required=False, help="number of hidden units in the first hidden layer")
ap.add_argument("--epochs", "-e", type=int, default=5, required=False, help="number of epochs the model will train to")
ap.add_argument("--gpu", "-g", action="store_true", help="If called model will run on gpu")
ap.add_argument("--save_dir", "-sd", required=False, help="subdirectory to save the model in")

args = vars(ap.parse_args())


data_dir = args['directory']
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

data_transforms = [train_transforms, valid_transforms, test_transforms]

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = data_transforms[0])
valid_data = datasets.ImageFolder(valid_dir, transform = data_transforms[1])
test_data = datasets.ImageFolder(test_dir, transform = data_transforms[2])

image_datasets = [train_data, valid_data, test_data]

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets[1], batch_size=64)
testloader = torch.utils.data.DataLoader(image_datasets[2], batch_size=64)

dataloaders = [trainloader, validloader, testloader]

if args["gpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = models.densenet121(pretrained = True)

if args["arch"] == "vgg16":
    model = models.vgg16(pretrained = True)
elif args["arch"] == "densenet121":
    model = models.densenet121(pretrained = True)
elif args["arch"] == None:
    print("Please use only resnet50 or densenet121, defaulting to densenet121")
    model = models.densenet121(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

    
    fc2out = (int)(args["hidden_units"] - ((args["hidden_units"] - 102)/2))
    fc3out = (int)(fc2out - ((fc2out - 102)/2))

    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, args["hidden_units"])
        self.fc2 = nn.Linear(args["hidden_units"], fc2out)
        self.fc3 = nn.Linear(fc2out, fc3out)
        self.fc4 = nn.Linear(fc3out, 102)
        
        self.dropout = nn.Dropout(p = .2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x
        
        
criterion = nn.NLLLoss()

model.classifier = Classifier()

optimizer = optim.Adam(model.classifier.parameters(), lr = args['learning_rate'])

model.to(device)


def save():
    epochs = 5
    model.class_to_idx = image_datasets[0].class_to_idx

    torch.save({"model_state_dict" : model.state_dict(),
                "class_to_idx" : model.class_to_idx,
                "Classifier" : model.classifier,
                "epoch" : epochs,
                "optimizer_state_dict" : optimizer.state_dict()}, "checkpoint.pth")

    
epochs = args['epochs']
steps = 0
print_every = 5
running_loss = 0
from workspace_utils import active_session


with active_session():
    for epoch in range(epochs):
        for inputs, labels in dataloaders[0]:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                accuracy = 0
                test_loss = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders[1]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(testloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
                
    save()
    