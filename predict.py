import os
import torch
from torch import nn
# path for data 
os.environ["SPEDAS_DATA_DIR"] = "/Volumes/datadir/mmsdata" #"/Users/naah5403/data"

import pyspedas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import more_itertools
import xarray as xr
from pytplot import tplot, del_data, options, get_data, get, store_data, ylim, tplot_options, tlimit
from pyspedas import tinterpol, time_string


def predictions(model:torch.nn.Module, 
          trange):

	tname = trange[0].replace('/','_').replace('-','').replace(':','')
	probe = '1'
	pyspedas.mms.fgm(trange=trange, data_rate='srvy', probe=probe)
	pyspedas.mms.fpi(trange=trange,center_measurement=True, data_rate='fast',datatype=['dis-moms'], probe=probe)
	pyspedas.mms.mec(trange=trange, data_rate='srvy', probe=probe)
	
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