import os
import torch
from torch import nn
# path for data 
os.environ["SPEDAS_DATA_DIR"] = "/Volumes/datadir/mmsdata"  #"/Users/naah5403/data"

import pyspedas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import more_itertools
from pyspedas import get_data
from pyspedas import tinterpol, time_string


def normalize_preserve_zero(tensor, min_val=None, max_val=None):
    """
    Normalize tensor to [-1, 1] while keeping zero at zero
    Uses piecewise linear transformation based on min/max values

    Args:
        tensor: Input tensor
        min_val: Minimum value (computed if None)
        max_val: Maximum value (computed if None)

    Returns:
        normalized tensor, min_val, max_val
    """
    if min_val is None:
        min_val = torch.min(tensor)
    if max_val is None:
        max_val = torch.max(tensor)

    # Get the absolute values of min and max for scaling
    abs_min = torch.abs(min_val)
    abs_max = torch.abs(max_val)

    # Initialize normalized tensor
    normalized = torch.zeros_like(tensor)

    # For negative values: map [min_val, 0] to [-1, 0]
    negative_mask = tensor < 0
    normalized[negative_mask] = tensor[negative_mask] / abs_min

    # For positive values: map [0, max_val] to [0, 1]
    positive_mask = tensor >= 0
    normalized[positive_mask] = tensor[positive_mask] / abs_max

    return normalized

def replace_pair_values(pred_tensor, pair_list, new_value):
    """
    Replace values in tensor based on pairs in pair_list

    Args:
        pred_tensor:
        pair_list: list of (a,b) pairs to consider (will treat (a,b) and (b,a) the same)
        new_value: scalar to set on both elements of each matched pair

    Returns:
        tensor with values replaced
    """
    # Build mask for pairs in y[:-1] vs y[1:]
    a = pred_tensor[:-1]
    b = pred_tensor[1:]
    # Start with all False
    mask_pairs = torch.zeros_like(a, dtype=torch.bool)
    for p in pair_list:
        p0, p1 = p
        mask_pairs |= ((a == p0) & (b == p1)) | ((a == p1) & (b == p0))

    # Expand to full-length mask marking both members of each matched pair
    mask_full = torch.zeros_like(tensor, dtype=torch.bool)
    mask_full[:-1] |= mask_pairs
    mask_full[1:] |= mask_pairs

    pred_tensor[mask_full] = new_value
    return pred_tensor

def predictions(model:torch.nn.Module,
          trange):

    tname = trange[0].replace('/','_').replace('-','').replace(':','')
    probe = '1'
    pyspedas.projects.mms.fgm(trange=trange, data_rate='srvy', probe=probe)
    pyspedas.projects.mms.fpi(trange=trange,center_measurement=True, data_rate='fast',datatype=['dis-moms'], probe=probe)
    pyspedas.projects.mms.mec(trange=trange, data_rate='srvy', probe=probe)

    BgseN ='mms1_fgm_b_gse_srvy_l2'
    DeniN = 'mms1_dis_numberdensity_fast'
    ViN = 'mms1_dis_bulkv_gse_fast'
    TiperpN = 'mms1_dis_tempperp_fast'
    TiparaN = 'mms1_dis_temppara_fast'
    posN = 'mms1_mec_r_gse'

    omni_flux = get_data('mms1_dis_energyspectr_omni_fast')
    energy = get_data('mms1_dis_energy_fast')

    # Interpol to ion fpi time 30ms
    tinterpol(BgseN, DeniN)
    tinterpol(posN, DeniN)

    #times, Bgse = get_data(BgseN+'-itrp')

    ni = get_data(DeniN)
    B = get_data(BgseN+'-itrp')
    Vi = get_data(ViN)
    Tiperp = get_data(TiperpN)
    Tipara = get_data(TiparaN)
    pos = get_data(posN+'-itrp')

    df_time = time_string(ni.times)
    df_ni = pd.DataFrame(ni.y, columns = ['ni'],index = [df_time])
    df_B = pd.DataFrame(B.y, columns = ['Bx','By','Bz','Btot'],index = [df_time])
    df_Vi = pd.DataFrame(Vi.y, columns = ['Vx','Vy','Vz'],index = [df_time])
    df_Vitot = pd.DataFrame(np.linalg.norm(df_Vi.values, axis=1), columns = ['vitot'], index = [df_time])
    df_Tiperp = pd.DataFrame(Tiperp.y, columns = ['Tiperp'],index = [df_time])
    df_Tipara = pd.DataFrame(Tipara.y, columns = ['Tipara'],index = [df_time])
    df_Ti = pd.concat([df_Tiperp,df_Tipara], axis=1)
    df_Titot = df_Ti.mean(axis=1)
    df_pos = pd.DataFrame(pos.y/6378., columns = ['X','Y','Z'],index = [df_time])

    df_final = pd.concat([df_B, df_ni, df_Vi, df_Vitot, df_Ti, df_Titot, df_pos], axis=1)
    df_final.columns = ['bx', 'by', 'bz', 'btot', 'ni', 'vix', 'viy', 'viz', 'vitot', 'Tiperp', 'Tipara', 'Titot', 'X', 'Y', 'Z']



    block_size = 40
    blocks = list(more_itertools.chunked(df_final.values, block_size))
    blocks = [np.array(x) for x in blocks]

    x_input_all = np.array(blocks[:-1])

    timestamps = list(more_itertools.chunked(df_final.index, block_size))
    timestamps = np.array(timestamps[:-1])

    index = pd.DatetimeIndex(timestamps[:,0,0])

    block_size = 40
    blocks = list(more_itertools.chunked(omni_flux.y, block_size))
    blocks = [np.array(x) for x in blocks]

    x_input_flux = np.array(blocks[:-1])


    X1 = x_input_flux

    X2_1 = x_input_all[:,:,3]  # btot
    X2_1 = np.expand_dims(X2_1, axis=2)
    X2_2 = x_input_all[:,:,11:13]  # T_tot, X

    X2 = np.concatenate((X2_1, X2_2), axis=2)

    import torch
    X1_tensor_test = torch.from_numpy(np.array(X1)).type(torch.float) # float is float32
    X2_tensor_test = torch.from_numpy(np.array(X2)).type(torch.float) # float is float32

    nan_mask = torch.isnan(X2_tensor_test)
    num_nan = torch.sum(nan_mask).item()
    print("Number of NaNs:", num_nan)

    # Replace NaN values with 0
    X2_tensor_test = torch.nan_to_num(X2_tensor_test, nan=0.0)

    min_vals_X1 = torch.tensor(0.)
    max_vals_X1 = torch.tensor(1.5701e+08)
    min_vals_X2 = torch.tensor([0.6576, 19.5559, -23.5244])
    max_vals_X2 = torch.tensor([85.0814, 9750.4180, 16.6774])

    PARAM_SIZE = 3

    X1_tensor_test = (X1_tensor_test - min_vals_X1) / ( max_vals_X1 - min_vals_X1)


    for i in range(0,PARAM_SIZE):
        X2_tensor_test[:,:,i] = (X2_tensor_test[:,:,i] - min_vals_X2[i]) / ( max_vals_X2[i] - min_vals_X2[i])


    # change to color, height, width, torch format
    X1_tensor_test = torch.permute(X1_tensor_test, (0, 2, 1))
    X1_tensor_test.size()

    X2_tensor_test = torch.permute(X2_tensor_test, (0, 2, 1))

    X1_tensor_test = X1_tensor_test.unsqueeze(dim=1)

    from tqdm.auto import tqdm

    # Make predictions with trained model
    num_samples = len(X1_tensor_test)
    batch_size = 8

    y_preds = []
    model.eval()


    with torch.inference_mode():
        for i in tqdm(range(0, num_samples, batch_size)):
            # print(i)
            batch_X1 = X1_tensor_test[i:i+batch_size]
            batch_X2 = X2_tensor_test[i:i+batch_size]

            y_logit = model(batch_X1, batch_X2)
            arr = torch.softmax(y_logit, dim=1)
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            top_two_values = []
            top_two_indices = []

            for row in arr:
                # Get indices of the two largest elements
                # print(row)
                idx = np.argsort(row)[-2:] #[::-1]  # Sort and take last two, reversed (second largest first)
                values = row[idx]
                top_two_values.append(values)
                top_two_indices.append(idx)

            # print(np.array(top_two_indices))

            y_preds.append(y_pred)

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)


    y_pred_tensor = pd.DataFrame(y_pred_tensor)
    y_pred_tensor = y_pred_tensor.set_index(index)

    figure, axis = plt.subplots(2, 1,  figsize=(20, 10), sharex=True, constrained_layout=True) #sharex=True
    pcm=axis[0].pcolormesh(np.array(time_string(energy.times)).astype("datetime64[ns]"),energy.y[0],
                                   np.log10(np.transpose(omni_flux.y[:,:])),cmap='nipy_spectral',shading='auto')
    axis[0].set_yscale('log')
    figure.colorbar(pcm, ax=axis[0], label="$keV/(cm^2~s~sr~keV)$", pad=0)
    # pcm.set_clim(3,6)

    size=14
    plt.rc('font', size=size)          # controls default text sizes
    plt.rc('axes', titlesize=size)     # fontsize of the axes title
    plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size+2)    # legend fontsize
    plt.rc('figure', titlesize=size)  # fontsize of the figure title


    axis[1].plot(index, y_pred_tensor, 'ok')
    axis[1].yaxis.set_ticks(np.arange(0,6,1))
    axis[1].yaxis.set_ticklabels(['SW', 'FS', 'MSH', 'MSP', 'PS', 'LOBE'])
    axis[1].grid()
    axis[1].set_ylabel('Predictions')
    # axis[1].set_xlabel('time (UTC)')


    # plt.savefig("FigCase_"+tname+".png")
    plt.show()


def predictions_cnn_rf(cnn_model: torch.nn.Module, rf_model, trange):

    probe = '1'
    pyspedas.projects.mms.fgm(trange=trange, data_rate='srvy', probe=probe, spdf=True)
    pyspedas.projects.mms.fpi(trange=trange, center_measurement=True, data_rate='fast', datatype=['dis-moms'],
                              probe=probe, spdf=True)
    pyspedas.projects.mms.mec(trange=trange, data_rate='srvy', probe=probe, spdf=True)

    BgseN = 'mms1_fgm_b_gse_srvy_l2'
    DeniN = 'mms1_dis_numberdensity_fast'
    ViN = 'mms1_dis_bulkv_gse_fast'
    TiperpN = 'mms1_dis_tempperp_fast'
    TiparaN = 'mms1_dis_temppara_fast'
    posN = 'mms1_mec_r_gse'

    omni_flux = get_data('mms1_dis_energyspectr_omni_fast')
    energy = get_data('mms1_dis_energy_fast')

    # Interpol to ion fpi time 30ms
    tinterpol(BgseN, DeniN)
    tinterpol(posN, DeniN)

    # times, Bgse = get_data(BgseN+'-itrp')

    ni = get_data(DeniN)
    B = get_data(BgseN + '-itrp')
    Vi = get_data(ViN)
    Tiperp = get_data(TiperpN)
    Tipara = get_data(TiparaN)
    pos = get_data(posN + '-itrp')

    df_time = time_string(ni.times)
    df_ni = pd.DataFrame(ni.y, columns=['ni'], index=[df_time])
    df_B = pd.DataFrame(B.y, columns=['Bx', 'By', 'Bz', 'Btot'], index=[df_time])
    df_Vi = pd.DataFrame(Vi.y, columns=['Vx', 'Vy', 'Vz'], index=[df_time])
    df_Vitot = pd.DataFrame(np.linalg.norm(df_Vi.values, axis=1), columns=['vitot'], index=[df_time])
    df_Tiperp = pd.DataFrame(Tiperp.y, columns=['Tiperp'], index=[df_time])
    df_Tipara = pd.DataFrame(Tipara.y, columns=['Tipara'], index=[df_time])
    df_Ti = pd.concat([df_Tiperp, df_Tipara], axis=1)
    df_Titot = df_Ti.mean(axis=1)
    df_pos = pd.DataFrame(pos.y / 6378., columns=['X', 'Y', 'Z'], index=[df_time])

    df_final = pd.concat([df_B, df_ni, df_Vi, df_Vitot, df_Ti, df_Titot, df_pos], axis=1)
    df_final.columns = ['bx', 'by', 'bz', 'btot', 'ni', 'vix', 'viy', 'viz', 'vitot', 'Tiperp', 'Tipara', 'Titot', 'X',
                        'Y', 'Z']

    block_size = 40
    blocks = list(more_itertools.chunked(df_final.values, block_size))
    blocks = [np.array(x) for x in blocks]

    x_input_all = np.array(blocks[:-1])

    timestamps = list(more_itertools.chunked(df_final.index, block_size))
    timestamps = np.array(timestamps[:-1])

    index = pd.DatetimeIndex(timestamps[:, 0, 0])

    block_size = 40
    blocks = list(more_itertools.chunked(omni_flux.y, block_size))
    blocks = [np.array(x) for x in blocks]

    x_input_flux = np.array(blocks[:-1])

    X1 = x_input_flux

    X2 = x_input_all[:, :, [3, 4, 5, 11, 12]]

    import torch
    X1_tensor_test = torch.log10(torch.from_numpy(np.array(X1)).type(torch.float) + 1.0)  # float is float32
    X2_tensor_test = torch.from_numpy(np.array(X2)).type(torch.float)  # float is float32

    nan_mask = torch.isnan(X2_tensor_test)
    num_nan = torch.sum(nan_mask).item()
    print("Number of NaNs:", num_nan)

    # Replace NaN values with 0
    X2_tensor_test = torch.nan_to_num(X2_tensor_test, nan=0.0)

    # normalization to training data values
    min_vals_X1 = torch.tensor(0.)
    max_vals_X1 = torch.tensor(8.4528)
    min_vals_X2 = torch.tensor([1.0475e-01, 5.6883e-02, -7.0503e+02, 1.1472e+01, -2.4599e+01])
    max_vals_X2 = torch.tensor([1.1128e+02, 2.2810e+02, 1.2067e+03, 1.1573e+04, 1.6951e+01])

    PARAM_SIZE = 5

    X1_tensor_test = (X1_tensor_test - min_vals_X1) / (max_vals_X1 - min_vals_X1)
    print('x1 shape : ', X1_tensor_test.shape)

    print('x2 shape : ', X2_tensor_test.shape)

    X2_tensor_test[:, :, 0] = (X2_tensor_test[:, :, 0] - min_vals_X2[0]) / (max_vals_X2[0] - min_vals_X2[0])
    X2_tensor_test[:, :, 1] = (X2_tensor_test[:, :, 1] - min_vals_X2[1]) / (max_vals_X2[1] - min_vals_X2[1])
    X2_tensor_test[:, :, 3] = (X2_tensor_test[:, :, 3] - min_vals_X2[3]) / (max_vals_X2[3] - min_vals_X2[3])

    X2_tensor_test[:, :, 2] = normalize_preserve_zero(X2_tensor_test[:, :, 2], min_vals_X2[2], max_vals_X2[2])
    X2_tensor_test[:, :, 4] = normalize_preserve_zero(X2_tensor_test[:, :, 4], min_vals_X2[4], max_vals_X2[4])

    # change to color, height, width, torch format
    X1_tensor_test = torch.permute(X1_tensor_test, (0, 2, 1))
    print('X1 after permute: ', X1_tensor_test.shape)

    X2_tensor_test = torch.permute(X2_tensor_test, (0, 2, 1))
    print('X2 after permute: ', X2_tensor_test.shape)

    X2_test_avg = np.array(X2_tensor_test.mean(axis=2))  # shape becomes (length, 3)
    print('X2 avg : ', X2_test_avg.shape)

    X1_tensor_test = X1_tensor_test.unsqueeze(dim=1)
    print('X1 after unsqueeze: ', X1_tensor_test.shape)

    # Make predictions with trained model
    from tqdm.auto import tqdm

    # Make predictions with trained CNN model and rf
    num_samples = len(X1_tensor_test)
    batch_size = 8

    y_pred_cnn = []
    cnn_model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_X1 = X1_tensor_test[i:i + batch_size]

            y_logit = cnn_model(batch_X1)
            y_cnn = torch.softmax(y_logit, dim=1)
            # print(y_cnn)
            y_pred_cnn.append(y_cnn)

    # # prediction from cnn model
    y_pred_cnn_tensor = torch.cat(y_pred_cnn)

    # prediction from random forest
    rf_output = rf_model.predict_proba(X2_test_avg)
    # prediction from cnn only
    y_cnn = np.argmax(y_pred_cnn_tensor.numpy(), axis=1)

    y_cnn = pd.DataFrame(y_cnn)
    y_cnn = y_cnn.set_index(index)

    # prediction from rf only
    y_rf = np.argmax(rf_output, axis=1)
    y_rf = pd.DataFrame(y_rf)
    y_rf = y_rf.set_index(index)

    # Fuse probablities (e.g., by averaging)

    combined_output = (y_pred_cnn_tensor.numpy() + rf_output) / 2
    y_pred_numpy = np.argmax(combined_output, axis=1)

    y_pred_tensor = torch.Tensor(y_pred_numpy)
    y_combined = y_pred_tensor
    y_combined = pd.DataFrame(y_combined)
    y_combined = y_combined.set_index(index)

    # Sequentially apply the same groups as your original code:
    # 1) (1<->2) -> 5  MSH <-> MSP  -> MP
    replace_pair_values(y_pred_tensor, [(1, 2)], 5)

    # 2) (0<->1) -> 6  SW <-> MSH -> BS
    replace_pair_values(y_pred_tensor, [(0, 1)], 6)

    # 3) (3<->4) -> 7 PS <-> LOBE -> PSBL
    replace_pair_values(y_pred_tensor, [(3, 4)], 7)

    y_pred_tensor = pd.DataFrame(y_pred_tensor)
    y_pred_tensor = y_pred_tensor.set_index(index)

    B = pd.DataFrame(X2_test_avg[:, 0])
    B = B.set_index(index)
    ni = pd.DataFrame(X2_test_avg[:, 1])
    ni = ni.set_index(index)
    vix = pd.DataFrame(X2_test_avg[:, 2])
    vix = vix.set_index(index)
    Ti = pd.DataFrame(X2_test_avg[:, 3])
    Ti = Ti.set_index(index)
    X = pd.DataFrame(X2_test_avg[:, 4])
    X = X.set_index(index)

    figure, axis = plt.subplots(5, 1, figsize=(20, 12), sharex=True, constrained_layout=True)  # sharex=True
    pcm = axis[0].pcolormesh(np.array(time_string(energy.times)).astype("datetime64[ns]"), energy.y[0],
                             np.log10(np.transpose(omni_flux.y[:, :])), cmap='nipy_spectral', shading='auto')
    axis[0].set_yscale('log')
    axis[0].set_ylabel('Ion Spectrogram')
    figure.colorbar(pcm, ax=axis[0], label="$keV/(cm^2~s~sr~keV)$", pad=0)
    # pcm.set_clim(3,6)

    size = 18
    plt.rc('font', size=size)  # controls default text sizes
    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size + 2)  # legend fontsize
    plt.rc('figure', titlesize=size)  # fontsize of the figure title

    axis[1].plot(index, B, 'ok', label='Magnetic Field')
    # axis[1].plot(index, Ni, 'og', label='Density')
    axis[1].plot(index, ni, 'og', label='Density')
    axis[1].plot(index, vix, 'oy', label='Velocity')
    axis[1].plot(index, Ti, 'or', label='Temperature')
    axis[1].plot(index, X, 'ob', label='Position')
    axis[1].grid()
    axis[1].set_ylabel('Parameters')
    axis[1].legend()

    axis[2].plot(index, y_rf, 'oc', label='RF')
    axis[2].plot(index, y_cnn, 'om', label='CNN')
    # axis[2].plot(index, y_pred_tensor, 'ok')
    # axis[2].yaxis.set_ticks(np.arange(0,9,1))
    # axis[2].yaxis.set_ticklabels(['SW', 'FS', 'MSH', 'MSP', 'PS', 'LOBE', 'MP', 'BS', 'PSBL'])
    axis[2].yaxis.set_ticks(np.arange(0, 5, 1))
    axis[2].yaxis.set_ticklabels(['SW', 'MSH', 'MSP', 'PS', 'LOBE'])
    axis[2].grid()
    axis[2].set_ylabel('RF and CNN')
    axis[2].legend()

    axis[3].plot(index, y_combined, 'og', label='Combined model')
    # axis[2].plot(index, y_pred_tensor, 'ok')
    # axis[2].yaxis.set_ticks(np.arange(0,9,1))
    # axis[2].yaxis.set_ticklabels(['SW', 'FS', 'MSH', 'MSP', 'PS', 'LOBE', 'MP', 'BS', 'PSBL'])
    axis[3].yaxis.set_ticks(np.arange(0, 5, 1))
    axis[3].yaxis.set_ticklabels(['SW', 'MSH', 'MSP', 'PS', 'LOBE'])
    axis[3].grid()
    axis[3].set_ylabel('Combined')
    axis[3].legend()

    axis[4].plot(index, y_pred_tensor, 'ok', label='Combined model with boundaries')
    axis[4].yaxis.set_ticks(np.arange(0, 8, 1))
    axis[4].yaxis.set_ticklabels(['SW', 'MSH', 'MSP', 'PS', 'LOBE', 'MP', 'BS', 'PSBL'])
    axis[4].grid()
    axis[4].set_ylabel('Predictions')
    axis[4].legend()
    # axis[1].set_xlabel('time (UTC)')

    plt.show()
