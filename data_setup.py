import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def min_max_scaling(tensor, feature_range=(0, 1)):
  """
  Scales a tensor to a specified range using min-max scaling.

  Args:
    tensor (torch.Tensor): The input tensor.
    feature_range (tuple, optional): The desired range for the scaled tensor. Defaults to (0, 1).

  Returns:
    torch.Tensor: The scaled tensor.
  """
  min_val = tensor.min()
  max_val = tensor.max()
  scaled_tensor = (tensor - min_val) / (max_val - min_val)

  #Adjust to the desired range
  scaled_tensor = scaled_tensor * (feature_range[1] - feature_range[0]) + feature_range[0]
  return scaled_tensor


def creating_data(batch_size: int):
	"""Creates training and testing Datasets, will change to dataloaders.

  Reads the h5 file containing data and creates training and testing datasets
  and normalize them.

  Args:
    ?: Path to data directory.

  Returns:
    X1_train, X2_train, X1_test, X2_test, y_train, y_test
  """

	batch_size = 8
	
	with h5py.File('training_data.h5', 'r') as f:
	    flux = f['flux'][:]
	    params = f['parameters'][:]
	    labels = f['labels'][:]
	
	class_names = ['sw', 'fs', 'msh', 'msp', 'ps', 'lobe']
	class_to_idx = {'sw': 0, 'fs': 1, 'msh': 2, 'msp': 3, 'ps': 4, 'lobe':5}
	
	PARAM_SIZE = params.shape[2]
	
	
	X1_tensor = torch.from_numpy(flux).type(torch.float) # float is float32
	X2_tensor = torch.from_numpy(params).type(torch.float) # float is float32
	y_tensor = torch.from_numpy(labels).type(torch.LongTensor).squeeze()
	
	nan_mask = torch.isnan(X2_tensor)
	num_nan = torch.sum(nan_mask).item()
	print("Number of NaNs:", num_nan)
	
	# Replace NaN values with 0
	X2_tensor = torch.nan_to_num(X2_tensor, nan=0.0)
	
	# change to color, height, width, torch format
	X1_tensor = torch.permute(X1_tensor, (0, 2, 1))
	X1_tensor.size()
	
	X2_tensor = torch.permute(X2_tensor, (0, 2, 1))
		
	# Split indices for training and testing
	train_indices, test_indices = train_test_split(range(len(X1_tensor)), test_size=0.2, random_state=42)
	
	# Use indices to split the datasets
	X1_train, X1_test = X1_tensor[train_indices], X1_tensor[test_indices]
	X2_train, X2_test = X2_tensor[train_indices], X2_tensor[test_indices]
	y_train, y_test = y_tensor[train_indices], y_tensor[test_indices]
	
	# normalization	
	min_vals_X1 = X1_train.min()
	max_vals_X1 = X1_train.max()
	X1_train = min_max_scaling(X1_train, feature_range=(0, 1))
	min_vals_X1, max_vals_X1
	X1_test = (X1_test - min_vals_X1) / (max_vals_X1 - min_vals_X1)
	
	min_vals_X2 = torch.zeros(PARAM_SIZE)
	max_vals_X2 = torch.zeros(PARAM_SIZE)
	
	for i in range(0,PARAM_SIZE):
	    min_vals_X2[i] = X2_train[:,i,:].min()
	    max_vals_X2[i] = X2_train[:,i,:].max()
	    # X2_tensor[:,:,i] = min_max_scaling(X2_tensor[:,:,i], feature_range=(-1, 1))
	    X2_train[:,i,:] = min_max_scaling(X2_train[:,i,:], feature_range=(0, 1))
	    X2_test[:,i,:] = (X2_test[:,i,:] - min_vals_X2[i]) / (max_vals_X2[i] - min_vals_X2[i])

	# add dimension 1 to image data	
	X1_train = X1_train.unsqueeze(dim=1)
	X1_test = X1_test.unsqueeze(dim=1)

	from torch.utils.data import DataLoader, TensorDataset

	assert len(X1_train) == len(X2_train) == len(y_train), "Train Datasets must have the same length"
	assert len(X1_test) == len(X2_test) == len(y_test), "Test Datasets must have the same length"

	train_dataset = TensorDataset(X1_train, X2_train, y_train)
	test_dataset = TensorDataset(X1_test, X2_test, y_test)

	# Create a DataLoader
	
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	train_dataloader , test_dataloader
	
	return train_dataloader, test_dataloader, class_names