# train/local_train.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim


def train_local(model, train_loader, device, local_epochs=1, lr=0.001):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()