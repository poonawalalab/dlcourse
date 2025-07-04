
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import mlpModel
from data import RegressionDataset,MLPDataLoader


if __name__ == "__main__":
    dataset = RegressionDataset('RegressionData.pkl',True)
    data_loader = MLPDataLoader(dataset)
    train = data_loader.train_dataloader()
    val = data_loader.val_dataloader()
    use_cuda:bool = torch.cuda.is_available()
    use_mps:bool = torch.mps.is_available()
    device = torch.device("cuda:0" if use_cuda else "mps" if use_mps else "cpu")
    model = mlpModel(6,[10,10],1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    epochs:int = 1000
    loss_function = nn.MSELoss()
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(epochs)):
        epoch_train_loss = []
        epoch_val_loss = []
        model.train()
        for input,target in train:
            input, target = input.to(device),target.to(device)
            optimizer.zero_grad()
            out = model.forward(input)
            loss = loss_function(out,target)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())

        model.eval()
        for input,target in val:
            input, target = input.to(device),target.to(device)
            with torch.no_grad():
                out = model.forward(input)
            loss = loss_function(out,target)
            epoch_val_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss).item())
        val_loss.append(np.mean(epoch_val_loss).item())
    
    all_val_inputs = dataset[:][0].to(device)
    all_val_predictions = model(all_val_inputs)
    all_val_targets = dataset[:][1]
    predictions_to_graph = all_val_predictions.cpu().detach().numpy().flatten()
    targets_to_graph = all_val_targets.numpy().flatten()


    fig,ax = plt.subplots(2)
    ax[0].plot(train_loss,c='b',label = 'Train')
    ax[0].plot(val_loss,c='r',label = 'Val')
    ax[0].legend()
    ax[0].set_title("Training Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].scatter(predictions_to_graph,targets_to_graph)
    ax[1].set_title("Results")
    ax[1].set_xlabel("Prediction")
    ax[1].set_ylabel("Target")
    plt.show()
