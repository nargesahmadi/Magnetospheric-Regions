import torch
from torch import nn
from torchmetrics import Accuracy

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #changed from 0.001
# setup metric and make sure it's on target device
acc_fn = Accuracy(task="multiclass", num_classes=6)

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
    In the form: {epochs: [...],
				  train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
  """
	results = {"epochs": [],
	           "train_loss": [],
	           "train_acc": [],
	           "test_loss": [],
	           "test_acc": []}
	
	# Training loop
	num_epochs = 20
	batch_size = 8
	num_samples = len(X1_train)
	for epoch in range(num_epochs):
	    train_loss, train_acc = 0, 0
	    # print(epoch)
	    for i in range(0, num_samples-1, batch_size):
	        batch_X1 = X1_train[i:i+batch_size]
	        batch_X2 = X2_train[i:i+batch_size]
	        batch_y = y_train[i:i+batch_size]
	
	        outputs = model(batch_X1, batch_X2)
	
	        loss = criterion(outputs, batch_y)
	        train_loss += loss
	        train_acc += acc_fn(outputs.argmax(dim=1), batch_y)
	        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}, acc: {train_acc.item()*100:.2f}")
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	    
	    train_loss /= (len(X1_train)/batch_size)
	    train_acc /= (len(X1_train)/batch_size)
	    print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {train_loss.item():.4f}: Train acc: {train_acc.item()*100:.2f}%")
	    results["epochs"].append(epoch)
	    results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
	    results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
	
	
	    test_loss, test_acc = 0, 0
	    # Put model into evaluation mode
	    model.eval()
	
	    with torch.inference_mode():
	        for i in range(0, len(X1_test), batch_size):
	            batch_X1 = X1_test[i:i+batch_size]
	            batch_X2 = X2_test[i:i+batch_size]
	            batch_y = y_test[i:i+batch_size]
	            # print(batch_y)
	
	            test_pred = model(batch_X1, batch_X2)
	            # print(test_pred.argmax(dim=1))
	            test_loss += criterion(test_pred , batch_y)
	            test_acc += acc_fn(test_pred.argmax(dim=1), batch_y)
	
	        # test loss and accuracy average per batch
	        test_loss /= (len(X1_test)/batch_size)
	        test_acc /= (len(X1_test)/batch_size)
	        print(f"\nEpoch [{epoch+1}/{num_epochs}]: Test loss: {test_loss:.4f}: Test acc: {test_acc*100:.2f}%\n")
	        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
	        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
	
	        # if test_acc > 0.98:  # Condition to exit loop
	        if test_loss < 0.05:  # Condition to exit loop
	            print("Stopping early: Accuracy exceeded threshold!")
	            break  # Exiting the loop
	print("Training complete!")
	  # Return the filled results at the end of the epochs
    return results