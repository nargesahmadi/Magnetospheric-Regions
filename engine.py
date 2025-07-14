import torch
from torch import nn
from torchmetrics import Accuracy

# setup metric and make sure it's on target device
acc_fn = Accuracy(task="multiclass", num_classes=6)


def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training with model traying to learn on data_loader"""
    
    train_loss, train_acc = 0, 0
    model.to(device)
    
    # Put model into training mode
    model.train()
    
    # Add a loop to loop through training batches
    for batch, (X1, X2, y) in enumerate(data_loader):
        
        # X1, X2, y = batch  # X1 from dataset1, X2 from dataset2, y as labels
        
        # Put data on target device
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        # print(X1.shape)
        # print(X2.shape)        
        # Transpose input to (batch_size, num_features, sequence_length)
        # X = X.transpose(1, 2)

        # 1. forward pass
        y_pred = model(X1, X2)
        # print('ypred=', y_pred)
        # 2. Calculate loss and accuracy (per patch)
        loss = loss_fn(y_pred, y)
        # print('loss =', loss)
        train_loss += loss # accumulate train loss
        train_acc += acc_fn(y_pred.argmax(dim=1), y) # (preds need to be same as y_true) going from logits -> y labels
        
        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss.backward()

        # 5. optimizer step
        optimizer.step()


    # Divide total train loss and total accuracy by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc*100:.2f}%")
    return train_loss, train_acc
    
def test_step(model: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             accuracy_fn,
             device: torch.device = device):

    test_loss, test_acc = 0, 0
    model.to(device)

    # Put model into evaluation mode
    model.eval()

    with torch.inference_mode():
        
        for batch, (X1, X2, y) in enumerate(data_loader):
            
            # X1, X2, y = batch  # X1 from dataset1, X2 from dataset2, y as labels
            # Put data on target device
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            # Transpose input to (batch_size, num_features, sequence_length)
            # X = X.transpose(1, 2)
            test_pred = model(X1, X2)
            # print(test_pred)
            test_loss += loss_fn(test_pred, y)
            test_acc += acc_fn(test_pred.argmax(dim=1), y)
            # print(test_pred.argmax(dim=1))
            # print(y)
        # test loss and accuracy average per batch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"\nTest loss: {test_loss:.4f}, Test acc: {test_acc*100:.2f}%\n")
        return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {epoch: [...],
				  train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
  """
  # Create empty results dictionary
  results = {"epoch": [],
	  "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["epoch"].append(epoch)
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results
