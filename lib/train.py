import numpy as np 

import torch 
from torch import nn, optim 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_progress(loss_trace, accuracy_trace, x, y, y_pred):
    clear_output(wait=True)
    
    fig, axis = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [2, 2, 1]})
    
    plt.subplot(1, 3, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(loss_trace)
                
    plt.subplot(1, 3, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_trace)
    
    plt.subplot(1, 3, 3)
#     img = x[0].detach().cpu().permute(1, 2, 0).numpy() / 2 + 0.5
#     plt.imshow(img)
    plt.axis('off')
    plt.title(f'true: {y[0]}    pred: {y_pred[0]}')
    
    plt.show()


def predict(pred):
    return torch.argmax(pred, axis=1)

def score(y_pred, y):
    N = len(y)
    return ((y_pred == y).sum() / N).item()
    
def train(model, train_dataloader, criterion, optimizer, n_epochs=5, show=False, verbose=10):
    
    model.train()
    loss_trace = []
    accuracy_trace = []
    
    for epoch_i in range(n_epochs):        
        for iter_i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)            
            pred = model(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())
            y_pred = predict(pred)
            accuracy_trace.append(score(y_pred, y))
            if show and (iter_i + 1) % verbose == 0:
                show_progress(loss_trace, accuracy_trace, x, y, y_pred)

    return loss_trace[-1], accuracy_trace[-1]


def test(model, test_dataloader):

    model.eval()
    total_accuracy = 0.0
    total = 0
    
    with torch.no_grad():
        for x, y in test_dataloader:            
            x, y = x.to(device), y.to(device)            
            pred = model(x)
            y_pred = predict(pred)
            total_accuracy += (y_pred == y).sum()
            total += len(y)
  
    total_accuracy /= total
    return total_accuracy.item()


def check(model, train_dataloader, test_dataloader, n_epochs=1, show=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loss, train_accuracy = train(model, train_dataloader, 
                                       criterion, optimizer, 
                                       n_epochs=n_epochs, show=show)
    
    test_accuracy = test(model, test_dataloader)
    paramount = model.params_amount()
    print(f'train loss: {train_loss:.3f}')
    print(f'train accuracy: {train_accuracy:.3f}')
    print(f'test  accuracy: {test_accuracy:.3f}')
    print(f'total params amount: {paramount}')
    return paramount, test_accuracy