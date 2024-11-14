

def baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch):
    model.train()

    train_loss = 0
    for i, data in enumerate(train_dataloader): 
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)

    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloder):
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    val_loss /= len(val_dataloder)
    
    return train_loss, val_loss
    