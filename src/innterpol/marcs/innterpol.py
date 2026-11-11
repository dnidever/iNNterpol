# -*- coding: utf-8 -*-
"""

MARCS 

Define marcs_inter_cae_nn() function to interpolate in values on a grid of model atmospheres
based on the MARCS dataset as described in Gustafsson B., Edvardsson B., Eriksson K., 
Jorgensen U.G., Nordlund A., Plez B. 2008, Astronomy & Astrophysics 486, 951
(https://doi.org/10.1051/0004-6361:200809724)

Uses a trained Neural Network (22 layers, fully connected) over the resulting encoding using a CAE with
a bottleneck of 71 elements. extracted over all the data.

Uses the hyperparameter file in .pth format (pyTorch) of both the NN and the CAE, together with the 
model.py containing the training model and model_encoder.py containing the model CAE. 

Autor: cwestend (https://github.com/cwestend/iNNterpol_MARCS)

"""

import numpy as np
import torch
from pathlib import Path
from . import model as network_model, model_encoder as encoder_model

_DATA = Path(__file__).resolve().parent

def parse_input(in_values):
    try:
        #in_values = np.array(in_values, dtype=float32) 
        in_val = np.copy(np.array(in_values, dtype=float))
        if len(in_val) != 5:
            raise ValueError("Incorrect size of input parameters")
        else:
        #    # Logg in 10x format
        #    in_val[4] = 10*in_val[4]
            return in_val
    except:
        raise ValueError("Need an array-like with 5 params: metalicity(log), carbon(log), other(log), temp, g(log)")

def innterpol(input_values):
    """ Function to interpolate in MARCS data: input 5 values of metal, carbon, other, temp, logg
        (array like) and outputs an array(56,4) with optical depth variation of mass(log), temp(log), 
        pres(log), and electronic density(log)"""

    inp_val = parse_input(input_values)
    # Rescale as trained (so they had similar size)
    inp_val[0] = inp_val[0]/2.5
    inp_val[3] = inp_val[3]/8000.
    inp_val[4] = inp_val[4]/5. 

    # Values for de-normalizing
    # obtained for each parameter by (i.e for temp):
    # maxt_all = slicelogt.max(axis=1, keepdims=True)
    # mint_all = slicelogt.min(axis=1, keepdims=True)
    # maxt = round(maxt_all.mean(),2)
    # mint = round(mint_all.mean(),2)  
    maxm = 2.19
    minm = -1.45
    maxt = 3.91
    mint = 3.44
    maxp = 4.9
    minp = 1.11
    maxe = 14.41 
    mine = 7.99

    # Optical depth (if needed)
    # 56 points: beware of rounding errors of arange!  
    taur = np.append(np.arange(-5.0,-3.0, 0.2), np.append(np.arange(-3.0, 1.0, 0.1),np.linspace(1.0, 2.0, 6)))
    # Keep only meaningful decimals
    taur = np.round(taur,2)

    # The interpolating NN
    device = "cpu"
    checkpoint = _DATA / "2022-10-25-11-08-03.pth"
    chk = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyperparameters = chk['hyperparameters']
    model = network_model.Network(hyperparameters).to(device)
    # model = model.MLP(hyperparameters).to(device)
    model.load_state_dict(chk['state_dict'])
    model.eval()

    # The CAE that encodes/decodes (model with CAE weights for encoding/decoding)
    checkpoint_enc = _DATA / "2022-10-19-08-54-49.pth_encoder_bak"
    chk_enc = torch.load(checkpoint_enc, map_location="cpu", weights_only=False)
    hyperparameters_enc = chk_enc['hyperparameters']
    model_encoder = encoder_model.Network(hyperparameters_enc).to(device)
    model_encoder.load_state_dict(chk_enc['state_dict'])
    model_encoder.eval()
    

    # Output of the NN
    # We only need the forward pass, so we do not accumulate gradients
    with torch.no_grad():
        # We transform the input from Numpy to PyTorch tensor
        inputs = torch.tensor(inp_val.astype('float32')).to(device)
        out_nn = model(inputs)

    # Format to apply decoder (and recover the parameter stratification)
    out_nn_t = out_nn.unsqueeze(1).unsqueeze(0)

    # Applying the decoder on the predicted values by the NN
    # We only need the forward pass, so we do not accumulate gradients
    with torch.no_grad():
        inputs = out_nn_t.to(device)
        out_de = model_encoder.decoder(inputs)

    # Convert to array
    # We bring the result to the CPU (if it was on the GPU) and transform to Numpy
    out_de_n = out_de.cpu().numpy()

    # Transform to m,t,p and e parameters in depth
    marcs_inter_cae_nn = np.empty((56, 4),dtype=object)  

    marcs_inter_cae_nn[:,0] = out_de_n[0,0,:]*(maxm-minm)+minm
    marcs_inter_cae_nn[:,1] = out_de_n[0,1,:]*(maxt-mint)+mint
    marcs_inter_cae_nn[:,2] = out_de_n[0,2,:]*(maxp-minp)+minp
    marcs_inter_cae_nn[:,3] = out_de_n[0,3,:]*(maxe-mine)+mine


    return marcs_inter_cae_nn, taur
