import os
import torch
import data_setup, engine, model_builder, utilities
import torch
from torch import nn


# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 8
#HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


# Create train/test dataloader and get class names as a list
train_dataloader, test_dataloader, class_names = data_setup.creating_data(batch_size = BATCH_SIZE)

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model_builder.MultiInputModel().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 

# Use train() by calling it from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

import utilities

# Save the model with help from utils.py
utilities.save_model(model=model,
                 target_dir="models",
                 model_name="01_going_modular.pth")
