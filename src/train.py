import gc
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import torch
from torchvision import transforms
from tqdm import tqdm


def train_one_epoch(model, device, data_loader, optimizer, criterion, description="Training", transformations=None):
    model.train()
    total_loss = 0.
    for batch_idx, (data, labels) in tqdm(enumerate(data_loader), desc=description, total=len(data_loader), leave=True, ncols=80):
        data, labels = data.to(device), labels.to(device)
        if transformations:
            data = transformations(data)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        del data, labels, outputs
        #torch.cuda.empty_cache()
        gc.collect()
        
    n_batches = len(data_loader)
    avg_loss = total_loss / n_batches
        
    return avg_loss

def test_one_epoch(model, device, data_loader, optimizer, criterion, description="Testing", transformations=None):
    model.eval()
    total_loss = 0.
    total_correct = 0.
    n_samples = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in tqdm(enumerate(data_loader), desc=description, total=len(data_loader), leave=True, ncols=80):
            data, labels = data.to(device), labels.to(device)
            if transformations:
                data = transformations(data)
            outputs = model(data)
            total_loss += criterion(outputs, labels).item()  # sum up batch total_loss
            predictions = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct += predictions.eq(labels.view_as(predictions)).sum().item()
            n_samples += data.size(0)
            del data, labels, outputs
            #torch.cuda.empty_cache()
            gc.collect()
    n_batches = len(data_loader)
    avg_loss = total_loss / n_batches
    accuracy = total_correct / n_samples
    return avg_loss, accuracy




def training_loop(model, device, data_loaders, optimizer, criterion, num_epochs, model_save_path, transformations=dict()):
    liveloss = PlotLosses(outputs=[MatplotlibPlot()])

    valid_loss_min, _ = test_one_epoch(
                            model, 
                            device, 
                            data_loaders["valid"], 
                            optimizer, 
                            criterion, 
                            description=f"Initial Validation", 
                            transformations=transforms.Compose(transformations.get("valid", []))
    )
    
    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(
            model, 
            device, data_loaders["train"], 
            optimizer, 
            criterion, 
            description=f"Epoch {epoch} Training", 
            transformations=transforms.Compose(transformations.get("train", [])) # + train_transforms
        )

        # Validation
        valid_loss, valid_accuracy = test_one_epoch(
            model, 
            device, data_loaders["valid"], 
            optimizer, 
            criterion, 
            description=f"Epoch {epoch} Validation", 
            transformations=transforms.Compose(transformations.get("valid", []))
        )

        logs = dict()
        logs["loss"] = train_loss
        logs["val_loss"] = valid_loss
        #logs["acc"] = train_accuracy
        logs["val_acc"] = valid_accuracy * 100
        liveloss.update(logs)
        liveloss.send()

        print ('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(epoch, num_epochs, train_loss, valid_loss, 100 * valid_accuracy))

        # If the validation loss decreases by more than 1%, save the model
        if (valid_loss_min - valid_loss) / valid_loss_min > 0.01:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            # Save the weights to save_path
            torch.save(model.cpu().state_dict(), model_save_path)
            model.to(device)

            valid_loss_min = valid_loss
            
    #torch.cuda.empty_cache()