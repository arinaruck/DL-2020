import torch


def train(config, model, optimizer, scheduler, early_stopping,
          train_loader, val_loader=None):
    device = config.device
    epochs = config.epochs
    clip = 15
    criterion = torch.nn.L1Loss()
    for epoch in range(epochs):
        train_loss = train_epoch(train_loader, model, optimizer, scheduler, device, criterion)
        val_loss = val_epoch(val_loader, model, optimizer, scheduler, early_stopping, device, criterion, epoch)
        print(f'epoch: {epoch}, train loss: {train_loss}, val_loss: {val_loss}')
        torch.save({
            'model_state_dict': model.state_dict(),
            }, 'latest_checkpoint.pt')

        if early_stopping.early_stop:
            print("Early stopping")
            break


def train_epoch(train_loader, model, optimizer, scheduler, device, criterion, grad_acum=1):
    model.train()
    tr_loss = 0
    tr_steps = 0
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        tr_steps += 1
        if (tr_steps % grad_acum) == 0:
            optimizer.step()
            optimizer.zero_grad()
    return tr_loss / tr_steps
                

@torch.no_grad()         
def val_epoch(val_loader, model, optimizer, scheduler, early_stopping, device, criterion, epoch):
    val_loss = 0
    val_steps = 0
    for batch in val_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        val_loss += loss.item()
        val_steps += 1
    b_sz = pred.shape[0]
    early_stopping(val_loss, model, optimizer, scheduler)
    scheduler.step(val_loss)
    return val_loss / val_steps
