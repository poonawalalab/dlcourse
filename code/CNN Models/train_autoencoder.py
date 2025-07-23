import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm
from models import CNNAutoencoder
from data import ImageMaskDataset, BasicDataLoader

if __name__ == "__main__":
    root_path = r"Welding_Image_Data"
    dataset = ImageMaskDataset(root_path)
    data_loader = BasicDataLoader(dataset,batch_size=64)
    train = data_loader.train_dataloader()
    val = data_loader.val_dataloader()

    use_cuda:bool = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CNNAutoencoder(2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    epochs:int = 25
    loss_function = nn.BCEWithLogitsLoss()
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = []
        epoch_val_loss = []
        model.train()
        for img,target in train:
            img,target = img.to(device),target.to(device)
            optimizer.zero_grad()
            out = model.forward(img)
            loss = loss_function(out,target)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item()) 

        model.eval()
        for img,target in val:
            img,target = img.to(device),target.to(device)
            with torch.no_grad():
                out = model.forward(img)
            loss = loss_function(out,target)
            epoch_val_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss).item())
        val_loss.append(np.mean(epoch_val_loss).item())
    
    idx_to_graph = sample(range(480),5)
    graph_imgs = torch.stack([dataset[i][0] for i in idx_to_graph])
    graph_masks = torch.stack([dataset[i][1] for i in idx_to_graph])
    graph_pred = model(graph_imgs.to(device)) 
    predictions_to_graph = graph_pred.cpu().detach()



    fig,ax = plt.subplots(5,4)
    for i, row in enumerate(ax):
        row[0].imshow(graph_imgs[i].squeeze(0),cmap= 'gray')
        row[0].axis("off")
        row[1].imshow(predictions_to_graph[i].squeeze(0),cmap= 'gray')
        row[1].axis("off")
        row[2].imshow(predictions_to_graph[i].squeeze(0)>0.0,cmap= 'gray')
        row[2].axis("off")
        row[3].imshow(graph_masks[i].squeeze(0),cmap= 'gray')
        row[3].axis("off")
    plt.show()
