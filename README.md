Magnetospheric Region Identifier

This code identifies 5 magnetospheric regions (Solar wind (SW), Magnetosheath (MSH), Inner Magnetosphere (MSP), Plasma sheet (PS) and Lobe (LOBE)) using Magnetospheric MultiScale (MMS) mission data.

The hybrid model uses ion omni flux timeseries and plasma parameters (total magnetic field, total ion temperature, and position in X GSE). The Convolutional Neural Network (CNN) model is trained on ion omni flux timeseries and Random Forest is used for plasma parameters. The final prediction is the mean probabilities from both model.

Predictions can be made using Colab_prediction.ipynb file by just giving it the desired time range.  
