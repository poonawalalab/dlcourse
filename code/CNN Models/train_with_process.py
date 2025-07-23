import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import CNNclassifierWithProcess
from data import ImageDataset, BasicDataLoader, ImageProcessDataset

if __name__ == "__main__":
    root_path = r"Welding_Image_Data"
    dataset = ImageProcessDataset(root_path)
    data_loader = BasicDataLoader(dataset,batch_size=64)
    train = data_loader.train_dataloader()
    val = data_loader.val_dataloader()

    use_cuda:bool = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CNNclassifierWithProcess(3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    epochs:int = 25
    loss_function = nn.MSELoss()
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = []
        epoch_val_loss = []
        model.train()
        for proc,img,target in train:

            proc,img,target = proc.to(device),img.to(device),target.to(device)
            optimizer.zero_grad()
            out = model.forward(proc,img)
            loss = loss_function(out,target)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item()) 

        model.eval()
        for proc,img,target in val:
            proc,img,target = proc.to(device),img.to(device),target.to(device)
            with torch.no_grad():
                out = model.forward(proc,img)
            loss = loss_function(out,target)
            epoch_val_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss).item())
        val_loss.append(np.mean(epoch_val_loss).item())
    
    all_val_proc = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    all_val_imgs = torch.stack([dataset[i][1] for i in range(len(dataset))]).to(device)
    all_val_predictions = model(all_val_proc,all_val_imgs)
    all_val_targets = torch.cat([dataset[i][2] for i in range(len(dataset))])
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
