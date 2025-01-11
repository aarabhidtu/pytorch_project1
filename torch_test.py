import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data", # path where the train/test data is stored
    train=True, # speciifies whether it is a training or test dataset 
    download=True, # downloads the data from the internet if it isnt available at root
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# we pass the dataset as an arguement to the DataLoader
batch_size=64
train_dataloader=DataLoader(training_data,batch_size=batch_size) # wrapping an iterable over our train dataset
test_dataloader=DataLoader(test_data,batch_size=batch_size)
for X,y in test_dataloader:
    print("Shape of X [N,C,H,W]:",X.shape)
    print("Shape of y:",y.shape,y.dtype)
    break

# creating models
device=(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device")

# define model 
class NeuralNetwork(nn.Module):
    def __init__(self): # special method that acts as a constructor. It is automatically called when you create an instance of a class
        # its purpose is to initialize the attributes of a class with specific values
        super().__init__() # super() allows you to call methods of the parent class from the child class 
        self.flatten=nn.Flatten() # flattening an input means to reshape into a flat, 1D tensor for each batch.
        self.linear_relu_stack=nn.Sequential( #ordered container of modules. all the data passes throgh the model in the specified order
            # Fully connected layers require the input to be a 1D tensor. hence we flatten in the previous step
            # Here each neuron captures a different aspect or feature of the input image
            nn.Linear(28*28,512), # represents a fully connected layer.28*28 is the size of the image
            # The ReLU is applied element-wise to the output of the first layer 
            # sets all -ve values to 0 while leaving the +ve values unchanges
            # non-linearity helps identify relationships in data that do not follow a straight line
            # they are able to capture dependencies and interactions between features that are complex
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()
        )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits
model=NeuralNetwork().to(device)    
print(model)

# to train a model, we need a loss function and an optimizer
loss_fn=nn.CrossEntropyLoss() # loss function that is used to train the model
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3) # optimizer that is used to update the weights of the model

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader,model,loss_fn,optimizer): # optimizer updates the model's weights to reduce the loss
    size=len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader): # features X and corresponding label y
        X,y=X.to(device),y.to(device)
        # Compute prediction error
        pred=model(X) # maps input to predictions
        loss=loss_fn(pred,y) # measures how far rhe model's predictions are from the ground truth. 
        # Backpropagation
        loss.backward() # computes the gradients of the loss with respect to the model's weights using backpropogation
        # gradients tell the optimizer how to adjust the weights to reduce loss
        optimizer.step() # the optimizer updates the model's weights using the gradients computed in the previous step
        optimizer.zero_grad() # clears the gradient stored in the optimizer for the next iteration
        if batch%100==0: # every 100 batches, logs the current loss and shows progress
            loss,current=batch,loss.item()
            print(f"loss: {loss:>7f} [{current:>5f}/{size:>5f}]")
        
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval() # sets the model to evaluation mode
    test_loss, correct=0,0
    with torch.no_grad(): # tells PyTorch that we do not need to store the computation graph
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# the training process is conducted over several iterations. 
# during each epoch, the model learns parameters to make better predictions. We print the model's accuracy and loss at each epoch
# the accuracy should increase and loss should decrease with every epoch
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# saving the model
# common way to save models is to serialize the internal state dictionary (containing the model parameters)
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# loading the model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth",weights_only=True))

# the model can now be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
