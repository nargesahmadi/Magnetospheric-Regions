

# Import data_setup.py
from . import data_setup

# Create train/test dataloader and get class names as a list
X1_train, X2_train, y_train, X1_test, X2_test, y_test = data_setup.creating_data()



import torch
from torch import nn
# Import model_builder.py
from . import model_builder

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model_builder.MultiInputModel()

# Import engine.py
from . import engine

# Use train() by calling it from engine.py
engine.train(...)

