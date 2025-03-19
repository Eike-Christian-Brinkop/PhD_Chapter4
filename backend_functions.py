# -*- coding: utf-8 -*-
"""
This published version was created on 19.03.2025

@author: Eike-Christian Brinkop
https://www.linkedin.com/in/eike-christian-brinkop-138158211/


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~  Please read this documentation before usage  ~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script contains the functions and classes used for the data pipeline 
    and for optimising the neural networks.

"""

####################################################################################################
####################################################################################################
####################################################################################################
#######################################   Backend Functions   ######################################
####################################################################################################
####################################################################################################
####################################################################################################

execution = False
##################
##################
###  Packages  ###
##################
##################
from matplotlib import pyplot as plt
import pandas as pd, numpy as np, random as rd, datetime as dt
import copy, time, os,json
from scipy.stats import rankdata
from calendar import monthrange as camo
import torch as to
from torch import nn
import threading as th

# for pytorch DDP training:
import torch.multiprocessing as tomp
from torch.nn.parallel import DistributedDataParallel as toDDP
from torch.distributed import init_process_group

import warnings
warnings.filterwarnings('ignore')
# import torch.nn.functional as F
device = "cuda" if to.cuda.is_available() else "cpu" # device = "cpu" # 
backend = "nccl" if to.distributed.is_nccl_available() else "gloo"
global_data_mode = "numpy"
devices = ["cpu"]
device_count = to.cuda.device_count()
if device == "cuda":
    global_data_mode = "torch"
    # to.cuda.set_device(0)
    devices = [to.device(f"cuda:{i}") for i in range(device_count)]
device = devices[0]

print(f"Using {device} device for training.")

shape012 = [0,0,0,0]
print_lock = th.Lock()

###########################################################################
###########################################################################
###########################################################################
#########################   Standard Functions:   #########################
###########################################################################
###########################################################################
###########################################################################

class timeit():
    def __init__(self,round_=2):
        self.round=round_
        self.stamp = -1
        self.time = [["start",time.time()]]
    def __call__(self,stamp = None):
        if stamp is None:
            self.stamp +=1
            stamp = self.stamp+0
        self.time.append([stamp,time.time()])
        return round(self.time[-1][1]-self.time[-2][1],self.round)
    def __format__(self):
        return self.__call__()
    def total(self):
        return self.time[-1][1]-self.time[0][1]
    def reset(self):
        self.stamp = -1
        self.time = [["start",time.time()]]
    def present(self):
        pass

def print_hint2(p,factor=1,space=2,sign="#", aspect_ratio = 16/9, width_field = None):
    height= (1+2*factor)*2
    width = (len(p)+2*space+2*factor)
    adjust = 0
    if width_field is None:
        if width/height<aspect_ratio:
            adjust = int((aspect_ratio/(width/height)*width-width+2)/2)
        print(("\n"+sign*(len(p)+2*(factor+space+adjust)))*factor+"\n"+sign*(factor+adjust),p,
              sign*(factor+adjust)+"\n"+(sign*(len(p)+2*(factor+space+adjust))+"\n")*factor,sep=" "*space)
    else:
        width_field_min = space*2+factor*2+len(p)
        if width_field_min > width:
            print("print_hint2: Minimum width field too low! Is",width_field, 
                  " and should be ",width_field_min,"!")
            return
        padding = width_field-len(p)-2*space
        padding_r = padding//2
        padding_l = padding-padding_r
        print("\n"+(width_field*sign+"\n")*factor+padding_l*sign+space*" "+p+\
              space*" "+padding_r*sign+"\n"+(width_field*sign+"\n")*factor)

def avg_1n(n):
    if n == 0:
        return 0
    return n*(n+1)/2

def avg_deviation(n_groups):
    double = 0
    if n_groups <=0:
        return 0
    nhalf = n_groups//2
    for i in range(nhalf):
        double += (avg_1n(i)+ avg_1n(n_groups - 1 -i))/n_groups
    if n_groups > 1:
        double*=2
    if n_groups%2 != 0:
        double += avg_1n((n_groups-1)/2)/n_groups*2
    return double/n_groups

def HML_deviation(n_groups):
    if n_groups <2:
        return 0
    return avg_1n(n_groups - 1)/n_groups

def QAA(MA,quan=10,rnd = 5,
        test_name = "test", 
        pred_name = "pred",
        date_entity = "month"):
    MA["qp"]=MA.reset_index().groupby(date_entity)[pred_name].\
        transform(lambda x:pd.qcut(x,quan,labels=False,duplicates="drop")).tolist()
    MA["qt"]=MA.reset_index().groupby(date_entity)[test_name].\
        transform(lambda x:pd.qcut(x,quan,labels=False,duplicates="drop")).tolist()
    MA1 = (MA[MA["qp"]==MA["qt"]].shape[0]/MA.shape[0]-1/quan)
    R = avg_deviation(quan)
    R_max = ((quan-1)*(quan)/2+quan//2)/quan
    MA2 = R-np.mean(abs(MA["qp"]-MA["qt"]))
    HMLMA1 = (MA[(MA["qp"]==MA["qt"]) & (MA["qt"].isin([0,int(quan-1)]))].shape[0]/\
              MA[MA["qt"].isin([0,int(quan-1)])].shape[0]-1/quan)
    R2 = avg_1n(quan-1)/quan
    R2_max = quan-1
    HMLMA2 = R2-np.mean(abs(MA[MA["qt"].isin([0,int(quan-1)])]["qp"]-\
                         MA[MA["qt"].isin([0,int(quan-1)])]["qt"]))
    if np.isnan(MA["qp"][MA.index[0]])==True:
        return 0,0,0,0
    if MA1 >=0:
        MA1/=(1-1/quan)
    else:
        MA1/=1/quan
    if MA2 <0:
        MA2/= (R-R_max)
    else:
        MA2/= R
    if HMLMA1 >=0:
        HMLMA1/=(1-1/quan)
    else:
        HMLMA1/=1/quan
    if HMLMA2 <0:
        HMLMA2/=(R2-R2_max)
    else:
        HMLMA2/=R2
    return round(MA1,rnd),round(MA2,rnd),round(HMLMA1,rnd),round(HMLMA2,rnd)

def month_end(date):
    return date[:7]+"-"+str(camo(int(date[:4]),int(date[5:7]))[1])

def month_shift2(months_start,shift=1,end=False):
    '''
    # Testing Parameters:
    months_start = '2023-01-10'
    shift=-22
    end=False
    '''
    day = int(months_start[-2:])
    month = int(months_start[5:7])+shift
    year = int(months_start[:4])
    if month <= 0:
        year_shift = (12-month)//12*-1
        month   += 12*-1*year_shift
        year    += year_shift
    elif month<=12:
        pass
    else:
        year    += (month-1)//12
        month   = (month-1)%12+1
    year_month = "{year:d}-{month:>02d}-{day:02d}".format(
        year = year,
        month= month,
        day = day)
    if end == True:
        year_month = month_end(year_month)
    else: 
        year_month = min(month_end(year_month),year_month)
    return year_month

def logx(number:float,
         base:float=np.exp(1)):
    return np.log(number)/np.log(base)

def isnumber(number):
    try:
        float(number)
        return True
    except:
        return False

class BatchNorm_self(nn.Module):
    def __init__(self,num_features):
        super().__init__() # init the base class
    def __call__(self,x):
        return x

def Tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

class act_DoubleTanh(nn.Module):
    '''
    Double wave activation function
    '''
    def __init__(self,scale_factor=1.75):
        super().__init__() # init the base class
        self.scale_factor=scale_factor
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        s = self.scale_factor
        return 0.5*(to.tanh(s*(input-2.5)+s/2*2.5)+to.tanh(s*(input+2.5)-s/2*2.5))

class act_ScaleSoftsign(nn.Module):
    '''
    Double wave activation function
    '''
    def __init__(self,scale_factor=1.0,multiplier=2):
        super().__init__() # init the base class
        self.scale_factor   = scale_factor
        self.multiplier     = multiplier
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return self.multiplier*input/(abs(input)+self.scale_factor)

################################################################################
################################################################################
########################  Model Architecture Containers  #######################
################################################################################
################################################################################
      
##################################################
#################  Custom Layers  ################
##################################################

class Layer_Sigmoid_decay(nn.Module):
    def __init__(self,size_in):
        super().__init__()
        self.size_in, self.size_out = size_in, size_in
        weights = to.ones(1)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = to.zeros(1)
        self.bias = nn.Parameter(bias)

    def forward(self, x, y):
        # import pdb; pdb.set_trace()
        # x = to.Tensor(np.random.normal(0,1,10)); print(x); y = to.tensor(np.random.uniform(0,1,1))
        x_processed = to.add(1,-1/(1+to.exp(-y*self.weights*8+self.weights*4-self.bias)))
        x_processed = to.mul(x_processed,x)
        return x_processed
    
    def __test__(self):
        a_time = 0
        for i in range(1000):
            x = to.Tensor(np.random.normal(0,1,self.size_in))
            y = to.Tensor([.32])
            b_time = time.time()
            z = self.forward(x,y)
            a_time+=time.time() - b_time
            if i == 500:
                print(x,y,z,sep="\n")
        print("Time for 1000 operations = {:7.5f}".format(a_time))
    
    
class Layer_Sigmoid_decay_multi(nn.Module):
    def __init__(self,size_in):
        super().__init__()
        self.size_in, self.size_out = size_in, size_in
        weights = to.ones(size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = to.zeros(size_in)
        self.bias = nn.Parameter(bias)

    def forward(self, x, y):
        # import pdb; pdb.set_trace()
        # x = to.Tensor(np.random.normal(0,1,10)); print(x); y = to.tensor(np.random.uniform(0,1,1))
        x_processed = to.add(1,-1/(1+to.exp(-y*self.weights*8+self.weights*4-self.bias)))
        x_processed = to.mul(x_processed,x)
        return x_processed
    
    def __test__(self):
        a_time = 0
        for i in range(1000):
            x = to.Tensor(np.random.normal(0,1,self.size_in))
            y = to.Tensor([.32])
            b_time = time.time()
            z = self.forward(x,y)
            a_time+=time.time() - b_time
            if i == 500:
                print(x,y,z,sep="\n")
        print("Time for 1000 operations = {:7.5f}".format(a_time))
    
##################################################
######### Structures and Model Containers ########
##################################################
    
def structure_analyzer_str(item,level=0,max_length = 20,max_depth=5,string_rep = ""):
    if isinstance(item,list):
        iterators = range(len(item))
        string_rep += "list"
    elif isinstance(item,dict):
        iterators = list(item.keys())
        string_rep += "dict"
    else:
        string_rep += "\n"+level*"\t"+str(type(item))
        return string_rep
    
    if level<max_depth:
        for iterator in iterators[:max_length]:
            if type(item[iterator]) in [str] or isnumber(item[iterator]):
                string_rep += "\n"+"\t"*(level+1)+"[{}]".format(str(iterator))+" "+\
                    str(item[iterator]) +" "+str(type(item[iterator]))
                continue
            string_rep += "\n"+"\t"*(level+1)+"[{}]".format(str(iterator))
            string_rep = structure_analyzer_str(item[iterator],level+1,max_length,max_depth,string_rep)
        if len(iterators)>max_length:
            string_rep+="\n"+level*"\t"+"Contents long"+str(len(iterators))
    else:
        return string_rep+"\n"+level*"\t"+"---too deep---"
    return string_rep

class Layers():
    activations = {
        "ELU":nn.ELU(), #check
        # "Harshrink": nn.Harshrink(), #no limitation no smoothness
        "Hardsigmoid":nn.Hardsigmoid(), #check 
        "Hardtanh":nn.Hardtanh(), #check
        "Hardswish":nn.Hardswish(), #check, but looks like a worse GELU
        "LeakyReLU":nn.LeakyReLU(0.1),# check
        # "LogSigmoid":nn.LogSigmoid(), #no
        # "ReLU":nn.ReLU(), #LeakyReLU
        "ReLU6":nn.ReLU6(), #check
        "SELU":nn.SELU(), #check
        # "GELU":nn.GELU(), # Mish
        "Sigmoid":nn.Sigmoid(), #check
        # "SiLU":nn.SiLU(), # Mish
        "Mish":nn.Mish(), #check
        "Softplus":nn.Softplus(),
        # "Softshrink":nn.Softshrink(), #Tanhshrink
        "Softsign":nn.Softsign(), #check
        "Tanh":nn.Tanh(), #check
        "Tanhshrink":nn.Tanhshrink(), #check
        "act_ScaleSoftsign":act_ScaleSoftsign(),
        "act_DoubleTanh":act_DoubleTanh(),
     }
    FFNN_norm = {
        "BatchNorm_self":BatchNorm_self,
        "BatchNorm1d":nn.BatchNorm1d,
        }
    CNN_norm = {
        "BatchNorm_self":BatchNorm_self,
        "BatchNorm2d":nn.BatchNorm2d,
        }
    layer_types = {
        "layer_sigmoid_decay":Layer_Sigmoid_decay}
    def __init__(self,structure:dict,verbose= 0):
        self.structure = structure
        self.verbose = verbose
        self.complete_structure(verbose)
    def complete_structure(self,verbose):
        pass
    def __repr__(self):
        return structure_analyzer_str(self.structure)
    def serialise(self,inplace=False):
        pars = copy.deepcopy(self.structure)
        if "FFNN_norm" in pars.keys():
            for key in self.FFNN_norm:
                if pars["FFNN_norm"] is self.FFNN_norm[key]:
                    pars["FFNN_norm"] = key
                    break
        if "CNN_norm" in pars.keys():
            for key in self.CNN_norm:
                if pars["CNN_norm"] is self.CNN_norm[key]:
                    pars["CNN_norm"] = key
                    break
        if "FFNN_sec" in pars.keys():
            for layer_n in range(len(pars["FFNN_sec"])):
                for key in self.activations.keys():
                    if isinstance(pars["FFNN_sec"][layer_n]["activation"],
                                  self.activations[key].__class__):
                        pars["FFNN_sec"][layer_n]["activation"] = key
                        break
        if "CNN_sec" in pars.keys():
            for layer_n in range(len(pars["CNN_sec"])):
                for key in self.activations.keys():
                    if isinstance(pars["CNN_sec"][layer_n]["activation"],
                                  self.activations[key].__class__):
                        pars["CNN_sec"][layer_n]["activation"] = key
                        break
        if "TNN" in pars.keys():
            for layer_n in range(len(pars["TNN"])):
                for key in self.activations.keys():
                    if isinstance(pars["TNN"][layer_n]["activation"],
                                  self.activations[key].__class__):
                        pars["TNN"][layer_n]["activation"] = key
        if inplace:
            self.structure = pars
            return None
        return pars
    @classmethod
    def deserialise(cls,pars):
        if "TNN" in pars.keys():
            for layer_n in range(len(pars["TNN"])):
                pars["TNN"][layer_n]["activation"] = \
                    cls.activations[pars["TNN"][layer_n]["activation"]]
        if "CNN_sec" in pars.keys():
            for layer_n in range(len(pars["CNN_sec"])):
                pars["CNN_sec"][layer_n]["activation"] = \
                    cls.activations[pars["CNN_sec"][layer_n]["activation"]]
        if "FFNN_sec" in pars.keys():
            for layer_n in range(len(pars["FFNN_sec"])):
                pars["FFNN_sec"][layer_n]["activation"] = \
                    cls.activations[pars["FFNN_sec"][layer_n]["activation"]]
        if "CNN_norm" in pars.keys():
            pars["CNN_norm"] = cls.CNN_norm[pars["CNN_norm"]]
        if "FFNN_norm" in pars.keys():
            pars["FFNN_norm"] = cls.FFNN_norm[pars["FFNN_norm"]]
        
        return pars
    @classmethod
    def from_dict(cls,pars):
        pars = cls.deserialise(pars)
        if "TNN" in pars.keys():
            return Layers_TNN_CNN_FFNN(pars)
        elif "CNN_sec" in pars.keys():
            return Layers_FFNN_CNN(pars)
        else:
            return Layers_FFNN(pars)
            

##################################################
###################### FFNN: #####################
##################################################

class Layers_FFNN(Layers):
    # def __init__(self,structure,verbose=1):
    #     super().__init__(structure,verbose)
    def complete_structure(self,verbose):
        if "lookback_dim" not in self.structure.keys():
            self.structure["lookback_dim"] = 1
        if type(self.structure["input_dim"]) in [tuple,list]:
            self.structure["input_dim"] = tuple([int(i) for i in self.structure["input_dim"]])
        else:
            self.structure["input_dim"] = int(self.structure["input_dim"])
        out_features = int(np.prod(self.structure["input_dim"]))
        for layer_index in range(len(self.structure["FFNN"])):
            in_features = copy.deepcopy(int(out_features))
            if verbose>0:
                print("""FFNN Layer {layer:2d}: receives {infeat:6d} input_features""".\
                format(layer = layer_index,
                       infeat = in_features))
            layer = self.structure["FFNN"][layer_index]
            layer["in_features"] = copy.deepcopy(in_features)
            self.structure["FFNN"][layer_index] = copy.deepcopy(layer)
            out_features = int(layer["out_features"])
        if "encoder_layer" not in self.structure.keys():
            self.structure["encoder_layer"] = None
        else: 
            self.structure["encoder_layer"] = len(self.structure["FFNN"])//2
        if "instrumented" not in self.structure.keys():
            self.structure["instrumented"] = False
        self.structure["FFNN_norm"] = self.structure["FFNN_norm"]
        self.structure["FFNN_in_dim"] = int(np.prod(self.structure["input_dim"]))
        if verbose>0:
            print("""FFNN Layer {layer:2d}: final output is {out_features:2d}""".\
                  format(layer = layer_index,
                         out_features=out_features))

FFNN_structure = {
        "input_dim":(5,60),
        "channels":1,
        "FFNN":[{"out_features":64},
                {"out_features":32},
                {"out_features":16},
                {"out_features":1},
                ],
        "FFNN_norm":BatchNorm_self,
        "FFNN_sec":[{"activation":nn.Tanh(),
                     "dropout":0.01},
                    {"activation":nn.Tanh(),
                     "dropout":0.01},
                    {"activation":nn.Tanh(),
                     "dropout":0.01}],
        }
FFNN_structure = Layers_FFNN(FFNN_structure)

class FFNN(nn.Module):
    def __init__(self,
                 structure:Layers_FFNN,
                 verbose = 0):
        ### super calls initialisation of superclass first. 
        ### References root class to initiate other functions
        # import pdb; pdb.set_trace()
        super().__init__() # initialises parent class
        self.model_classification = "RN"
        self.structure = structure.structure
        if "seed" in self.structure.keys():
            to.manual_seed(self.structure["seed"])
        else:
            to.manual_seed(np.random.randint(0,10000))
        self.FFNN_sec = self.structure["FFNN_sec"]
        self.lookback_dim = self.structure["lookback_dim"]
        self.FFNN_norm = self.structure["FFNN_norm"](self.structure["FFNN_in_dim"]).double()
        self.lin = nn.ModuleList()
        self.dout = nn.ModuleList()
        self.verbose = verbose
        for layer in self.FFNN_sec:
            self.dout.append(nn.Dropout(layer["dropout"])).double()
        for FFNN_layer in self.structure["FFNN"]:
            self.lin.append(nn.Linear(**FFNN_layer).double())
    def forward(self, x, objective = None,**kwargs):
        if len(x.shape)>2:
            x = x[...,:self.lookback_dim]
        x = to.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.FFNN_norm(x)
        for layer in range(len(self.lin)-1):
            x = self.dout[layer](x)
            try:
                x = self.FFNN_sec[layer]["activation"](self.lin[layer](x))
            except TypeError: 
                print(x.shape,type(x),"\n",sep="\n")
                raise
        x = self.lin[-1](x)
        if objective is not None:
            x = objective.loss_fn.norm_pred(x)
        return x

##################################################
##################  Autoencoder  #################
##################################################

class AE_1d(nn.Module):
    def __init__(self,
                 structure:Layers_FFNN,
                 verbose = 0):
        ### super calls initialisation of superclass first. 
        ### References root class to initiate other functions
        self.model_classification = "AE"
        super().__init__() # initialises parent class
        self.structure = structure.structure
        self.FFNN_sec = self.structure["FFNN_sec"]
        self.FFNN_norm = self.structure["FFNN_norm"](self.structure["FFNN_in_dim"]).double()
        self.lin = nn.ModuleList()
        self.instrumented = self.structure["instrumented"]
        self.encoder_layer = self.structure["encoder_layer"]
        self.dout = nn.ModuleList()
        self.verbose = verbose
        for layer in self.FFNN_sec:
            self.dout.append(nn.Dropout(layer["dropout"]))
        for FFNN_layer in self.structure["FFNN"]:
            self.lin.append(nn.Linear(**FFNN_layer).double())
    def encoder(self, x, test=False):
        x = self.FFNN_norm(x)
        x = to.flatten(x, 1) # flatten all dimensions except the batch dimension
        for layer in range(self.encoder_layer):
            if not test:
                x = self.dout[layer](x)
            try:
                x = self.FFNN_sec[layer]["activation"](self.lin[layer](x))
            except TypeError: 
                print(x.shape,type(x),"\n",sep="\n")
                raise
        return x
    def decoder(self,x, test=False):
        for layer in range(self.encoder_layer,len(self.lin)-1):
            if not test:
                x = self.dout[layer](x)
            try:
                x = self.FFNN_sec[layer]["activation"](self.lin[layer](x))
            except TypeError: 
                print(x.shape,type(x),"\n",sep="\n")
                raise
        x = self.lin[-1](x)
        return x
    def forward(self, x, objective=None,**kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        x = to.unsqueeze(x,2)
        if objective is not None:
            x = objective.loss_fn.norm_pred(x)
        return x


##################################################
###################### CNN: ######################
##################################################

class Layers_FFNN_CNN(Layers):
    standard_pooling = 1
    def get_output_shape(self,input_dim,layer,reshape,verbose = 0):
        # import pdb; pdb.set_trace();
        output_shape = []
        reshape_applied = False
        for index in range(len(input_dim)):
            kernel_dim = layer["kernel_size"][index]
            dim = (input_dim[index]-(kernel_dim+(kernel_dim-1)*\
                                     (layer["dilation"][index]-1))+\
                   2*layer["padding"][index])/layer["stride"][index]+1
            if dim == int(dim):
                dim = int(dim)
            if dim == 1 and reshape and not reshape_applied:
                dim*=layer["out_channels"]
                reshape_applied=True
            if dim <1:
                print("Exception: class Layers_FFNN_CNN:get_ouput_shape: ")
                print("dim=",dim)
                print(kernel_dim, input_dim, layer, reshape,sep=",")
                raise
            output_shape.append(dim)
        return output_shape
    def complete_structure(self,verbose = 0):
        # import pdb; pdb.set_trace();
        if "lookback_dim" not in self.structure.keys():
            self.structure["lookback_dim"] = self.structure["input_dim"][1]
        output_dim = list(self.structure["input_dim"])
        output_dim[1] = self.structure["lookback_dim"]
        self.structure["CNN_out_dim"] = []
        if "CNN_pooling" not in self.structure.keys():
            self.structure["CNN_pooling"] = [tuple([self.standard_pooling+0]*len(output_dim))]*\
                len(self.structure["CNN"])
        out_channels = self.structure["in_channels"]+0
        # determine reshape after each cnn layer
        if "reshape" not in self.structure.keys():
            self.structure["reshape"] = [False]*len(self.structure["CNN"])
        # self.structure["CNN_norm"] = self.structure["CNN_norm"](output_dim[0])
        while len(self.structure["CNN"])>len(self.structure["CNN_sec"]):
            self.structure["CNN_sec"].append(self.structure["CNN_sec"])[-1]
        for layer in range(len(self.structure["CNN"])):
            if "dropout" not in self.structure["CNN_sec"][layer].keys():
                self.structure["CNN_sec"][layer]["dropout"] = 0
        reshape = 1
        for layer_index in range(len(self.structure["CNN"])):
            # layer_index = 1
            input_dim = output_dim+[]
            if verbose>0: 
                print("""CNN Layer {layer:2d}: receives input shape of {insh:15s}""".\
                      format(layer = layer_index,
                             insh = str((*input_dim,out_channels))))
            layer = self.structure["CNN"][layer_index]
            # extending the parameters to the length if input_dim #
            for characteristic in ["kernel_size","stride","padding","dilation"]:
                if isinstance(layer[characteristic], int):
                    value0 = layer[characteristic]+0
                    layer[characteristic] = tuple([layer[characteristic]]*len(input_dim))
                    if verbose>0:
                        print("""CNN Layer {layer:2d}: Changed {char:15s} 
                              from {value0:5d} to {value1:15s}""".\
                            format(layer= layer_index,
                                   char = characteristic,
                                   value0 = value0,
                                   value1 = str(layer[characteristic])))
            reshape = self.structure["reshape"][layer_index]
            # determine dimension of output
            output_dim = self.get_output_shape(
                input_dim, layer, reshape=reshape, verbose = verbose)#
            self.structure["CNN_out_dim"].append(output_dim)
            # wether to map channels into emptied dimension
            if not reshape:
                layer["in_channels"] = copy.deepcopy(out_channels)
            out_channels = layer["out_channels"]
            # correction of pooling layers:
            for dim in range(len(output_dim)):
                # dim = 0
                if output_dim[dim] % self.structure["CNN_pooling"][layer_index][dim] != 0:
                    print("found pooling layer {} and output_dim {}".format(
                       self.structure["CNN_pooling"][layer_index][dim],output_dim[dim]))
                    init_pooling = self.structure["CNN_pooling"][layer_index][dim]
                    while output_dim[dim] % self.structure["CNN_pooling"][layer_index][dim] != 0:
                        self.structure["CNN_pooling"][layer_index][dim]-=1
                        if self.structure["CNN_pooling"][layer_index][dim] <= 1:
                            # print("WARNING: found pooling layer ")
                            self.structure["CNN_pooling"][layer_index][dim] = 1
                            break
                    print("""CNN Layer {layer:2d}: Changed pooling in dim {dim:2d} 
                              from {value0:3d} to {value1:3d}""".\
                            format(layer= layer_index,
                                   dim = dim,
                                   value0 = init_pooling,
                                   value1 = self.structure["CNN_pooling"][layer_index][dim]))
                output_dim[dim]/=self.structure["CNN_pooling"][layer_index][dim]
                output_dim[dim] = int(output_dim[dim])
            if output_dim[dim] <1:
                for key in locals():
                    print(key,locals()[key])
                print("structure:",self.structure)
                raise
        out_features = np.prod((*output_dim,out_channels**(1-reshape)))
        if verbose>0: 
            print("""CNN Layer {layer:2d}: final   output_shape is {outsh:15s}
                  flattened to {n_neurons:6d}""".\
                format(layer = layer_index,
                       outsh = str((*output_dim,out_channels)),
                       n_neurons = out_features))
        if "FFNN_norm" not in self.structure.keys():
            self.structure["FFNN_norm"] = BatchNorm_self
        # self.structure["FFNN_norm"] = self.structure["FFNN_norm"](
        #     np.prod([*output_dim,out_channels]))
        self.structure["FFNN_in_dim"] = int(out_features)
        for layer_index in range(len(self.structure["FFNN"])):
            in_features = int(out_features)
            if verbose>0: print("""FFNN Layer {layer:2d}: receives {infeat:6d} input_features""".\
                    format(layer = layer_index,
                           infeat = in_features))
            layer = self.structure["FFNN"][layer_index]
            layer["in_features"] = copy.deepcopy(in_features)
            out_features = layer["out_features"]
        if verbose>0: 
            print("""FFNN Layer {layer:2d}: final output is {out_features:2d}""".\
                  format(layer = layer_index,
                         out_features=out_features))

CNN_v1_structure = {
        "input_dim":(5,60),
        "in_channels": 1, 
        "CNN":[{"in_channels":0,
                "out_channels":3,
                "kernel_size":(5,12),
                "stride":(2,4),
                "padding":(2,6),
                "dilation":1},
               {"in_channels":0,
                "out_channels":6,
                "kernel_size":(5,12),
                "stride":(1,2),
                "padding":(2,6),
                "dilation":1},],
        "CNN_sec":[
            {"activation":nn.Softsign()},
            {"activation":nn.Softsign()}
            ],
        "FFNN":[{"out_features":72},
                {"out_features":36},
                {"out_features":12},
                {"out_features":1},
                ],
        "CNN_norm":BatchNorm_self,
        "FFNN_sec":[{"activation":nn.Softsign(),
                     "dropout":0.0},
                    {"activation":nn.Softsign(),
                     "dropout":0.0},
                    {"activation":nn.Softsign(),
                     "dropout":0.0}],
        }
CNN_v1_structure = Layers_FFNN_CNN(CNN_v1_structure)

class CNN_v1(nn.Module):
    def __init__(self,
                 structure:Layers_FFNN_CNN,
                 verbose = 0,
                 ):
        ### super calls initialisation of superclass first. 
        ### References root class to initiate other functions
        super().__init__() # initialises parent class
        self.model_classification = "RN"
        self.structure = structure.structure
        self.conv = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.lookback_dim = self.structure["lookback_dim"]
        self.CNN_norm = self.structure["CNN_norm"](self.structure["in_channels"])
        self.FFNN_norm = self.structure["FFNN_norm"](self.structure["FFNN_in_dim"]).double()
        for pooling,CNN_layer in zip(self.structure["CNN_pooling"],self.structure["CNN"]):
            try:
                self.conv.append(nn.Conv2d(**CNN_layer).double())
            except TypeError:
                print("\nClass CNN_v1: One or more items in layer are unknown to pytorch Conv2d:\n")
                print(CNN_layer)
                raise
            if len(CNN_layer["kernel_size"])==2:
                self.pooling.append(nn.MaxPool2d(tuple(pooling)))
            elif len(CNN_layer["kernel_size"])==3:
                self.pooling.append(nn.MaxPool3d(tuple(pooling)))
        self.FFNN_sec = self.structure["FFNN_sec"]
        self.CNN_sec = self.structure["CNN_sec"]
        self.lin = nn.ModuleList()
        self.dout = nn.ModuleList()
        self.CNN_dout = nn.ModuleList()
        self.verbose = verbose
        self.check = True
        for layer in self.FFNN_sec:
            self.dout.append(nn.Dropout(layer["dropout"]))
        for layer in self.CNN_sec:
            self.CNN_dout.append(nn.Dropout(layer["dropout"]))
        for FFNN_layer in self.structure["FFNN"]:
            self.lin.append(nn.Linear(**FFNN_layer).double())
    def forward(self, x, objective = None,**kwargs):
        if len(x.shape)>2:
            x = x[...,:self.lookback_dim]
        if len(x.shape) == 3:
            x = x.view(-1,1,self.structure["input_dim"][0],self.lookback_dim)
        if self.verbose>0 and self.check:
            global shape012 
            shape012[0]= to.isnan(x).sum()
        x = self.CNN_norm(x)
        if self.verbose>0 and self.check:
            shape012[1] = to.isnan(x).sum()
            self.check = False
        for layer in range(len(self.conv)):
            x = self.CNN_dout[layer](x)
            x = self.CNN_sec[layer]["activation"](self.conv[layer](x))
            x = self.pooling[layer](x)
        x = to.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.FFNN_norm(x)
        for layer,secs in zip(range(len(self.lin)-1),self.FFNN_sec):
            x = self.dout[layer](x)
            x = self.lin[layer](x)
            x = secs["activation"](x)
        x = self.lin[-1](x)
        if objective is not None:
            x = objective.loss_fn.norm_pred(x)
        return x

class TNN_pos_emb():
    """
    produces an object of positional econding for a matrix.
    ––––––––––––––––––––––––––––––––––––
    ––––––––––––––––––––––––––––––––––––
    Parameters:
    ––––––––––––––––––––––––––––––––––––
    d: int, number of encodings for each word.
    l: int, number of words in sentence.
    nwavelength: int, scalar for scaling of positional encoding
    """
    def __init__(self,d,l,wavelength=0,magnitude = 1):
        transposed = False
        if wavelength <0 :
            transposed=True
            wavelength = -wavelength
            d_i = l+0
            l = d+0
            d = d_i
        matrix = np.zeros([1,l,d])
        if wavelength != 0:
            for d_i in range(d):
                i = d_i // 2+1
                if d_i%2==0:
                    func = np.sin
                else:
                    func = np.cos
                for k in range(l):
                    matrix[0,k,d_i] = func((k+1)/(wavelength**(2*i/d)))*magnitude
        self.matrix = matrix
        if transposed:
            self.matrix = np.transpose(matrix,[0,2,1])
    def get(self, n):
        return to.tensor(np.repeat(self.matrix,n,axis=0),device=device)
    def plot(self,color = "white"):
        plt.figure(figsize=(6, 6),facecolor=color)
        plt.imshow(np.squeeze(self.matrix),cmap='hot',interpolation = "nearest", aspect='auto')
        plt.colorbar()
        plt.show()

class Layers_TNN_CNN_FFNN(Layers_FFNN_CNN):
    def __init__(self,structure,verbose= 0):
        super().__init__(structure,verbose)
        self.TNN_structure(structure)
    def TNN_structure(self,structure):
        self.structure["TNN"] = structure["TNN"]
        if "wavelength" not in self.structure.keys():
            self.structure["wavelength"] = 0
        else:
            self.structure["wavelength"] = structure["wavelength"]

TNN_test_structure = CNN_v1_structure.structure
TNN_substructure = [
    {"d_model":20,
     "nhead":5,
     "dim_feedforward": 256,
     "dropout":0.1,
     "activation":"relu"},
    {"d_model":20,
     "nhead":5,
      "dim_feedforward": 256,
      "dropout":0.1,
      "activation":"relu"},
    ]
TNN_test_structure["TNN"] = TNN_substructure
TNN_test_structure = Layers_TNN_CNN_FFNN(TNN_test_structure)

# layers based on 312 inputs for reducing dimensions with a CNN unit and then applying TNN with fewer factors:
CNN_TNN_PhD4_layers = None

# layers based on 94 input variables for PhD calculations example:
CNN_TNN_PhD_layers = {
    "input_dim":(94,24),
    "in_channels":1,
    "CNN":[{"in_channels":1,
            "out_channels":4,
            "kernel_size":(16,8),
            "stride":(8,2),
            "padding":(5,2),
            "dilation":1},
           {"in_channels":4,
            "out_channels":8,
            "kernel_size":(12,8),
            "stride":(4,1),
            "padding":(4,3),
            "dilation":1},],
    "CNN_sec":[
        {"activation":nn.Softsign()},
        {"activation":nn.Softsign()}
        ],
    "FFNN":[{"out_features":80},
            {"out_features":40},
            {"out_features":1},
            ],
    "CNN_norm":BatchNorm_self,
    "FFNN_norm":BatchNorm_self,
    "FFNN_sec":[{"activation":nn.Softsign(),
                 "dropout":0.02},
                {"activation":nn.Softsign(),
                 "dropout":0.02},],
    }
CNN_TNN_PhD_layers = Layers_FFNN_CNN(CNN_TNN_PhD_layers)
CNN_TNN_PhD_layers = CNN_TNN_PhD_layers.structure
TNN_substructure = [
    {"d_model":24,
     "nhead":6,
     "dim_feedforward": 32,
     "dropout":0.2,
     "activation":nn.Softsign()},
    {"d_model":24,
     "nhead":6,
     "dim_feedforward": 32,
      "dropout":0.1,
      "activation":nn.Softsign()},
    ]
CNN_TNN_PhD_layers["TNN"] = TNN_substructure
CNN_TNN_PhD_layers = Layers_TNN_CNN_FFNN(CNN_TNN_PhD_layers)


class TNN_CNN(nn.Module):
    def __init__(self,
                 structure:Layers_TNN_CNN_FFNN, 
                 verbose = 0,
                 ):
        super().__init__() # initialises parent class
        self.model_classification = "RN"
        self.structure = structure.structure
        self.lookback_dim = self.structure["lookback_dim"]
        self.trans_pos_emb = TNN_pos_emb(*self.structure["input_dim"],self.structure["wavelength"])
        self.trans = nn.ModuleList()
        self.pooling = nn.ModuleList()
        for TNN_layer in self.structure["TNN"]:
            try:
                self.trans.append(nn.TransformerEncoderLayer(**TNN_layer,
                                                             batch_first = True).double())
            except TypeError:
                print("\nClass CNN_v1: One or more items in layer are unknown to pytorch Conv2d:\n")
                print(TNN_layer)
                raise
        self.conv = nn.ModuleList()
        self.norm = self.structure["FFNN_norm"](self.structure["FFNN"][0]["in_features"])
        for pooling,CNN_layer in zip(self.structure["CNN_pooling"],self.structure["CNN"]):
            try:
                self.conv.append(nn.Conv2d(**CNN_layer).double())
            except TypeError:
                print("\nClass TNN_CNN: One or more items in layer are unknown to pytorch Conv2d:\n")
                print(CNN_layer)
                raise
            if len(CNN_layer["kernel_size"])==2:
                self.pooling.append(nn.MaxPool2d(tuple(pooling)))
            elif len(CNN_layer["kernel_size"])==3:
                self.pooling.append(nn.MaxPool3d(tuple(pooling)))
        self.FFNN_sec = self.structure["FFNN_sec"]
        self.CNN_sec = self.structure["CNN_sec"]
        self.lin = nn.ModuleList()
        self.dout = nn.ModuleList()
        self.CNN_dout = nn.ModuleList()
        self.verbose = verbose
        self.check = True
        for layer in self.FFNN_sec:
            self.dout.append(nn.Dropout(layer["dropout"]))
        for layer in self.CNN_sec:
            self.CNN_dout.append(nn.Dropout(layer["dropout"]))
        for FFNN_layer in self.structure["FFNN"]:
            self.lin.append(nn.Linear(**FFNN_layer).double())
    def forward(self, x, objective = None,**kwargs):
        # import pdb; pdb.set_trace();
        if len(x.shape)>2:
            x = x[...,:self.lookback_dim]
        s_x = x.shape
        if self.verbose>0 and self.check:
            global shape012 
            shape012[0]= to.isnan(x).sum()
            shape012[1] =s_x
            shape012[2] = x.shape
            shape012[3] = x
        x = x.reshape(s_x[0],s_x[2],s_x[1])
        pos_emb = self.trans_pos_emb.get(x.shape[0]) #positional embedding
        for TNN_layer in self.trans:
            x += pos_emb
            x = TNN_layer(x)    
        del pos_emb
        if self.verbose>0 and self.check:
            shape012[1] = to.isnan(x).sum()
            self.check = False
        x = x.reshape(*s_x)
        if len(x.shape) == 3:
            x = x.view(-1,1,self.structure["input_dim"][0],self.lookback_dim)

        for layer in range(len(self.conv)):
            x = self.CNN_dout[layer](x)
            x = self.CNN_sec[layer]["activation"](self.conv[layer](x))
            x = self.pooling[layer](x)
            if self.structure["reshape"][layer]:
                x = x.view(x.shape[0],1,*self.structure["CNN_out_dim"][layer])
        x = to.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.norm(x)
        for layer,secs in zip(range(len(self.lin)-1),self.FFNN_sec):
            x = self.dout[layer](x)
            x = self.lin[layer](x)
            x = secs["activation"](x)
        x = self.lin[-1](x)
        if objective is not None:
            x = objective.loss_fn.norm_pred(x)
        return x

class CNN_TNN(nn.Module):
    def __init__(self,
                 structure:Layers_TNN_CNN_FFNN, 
                 verbose = 0,
                 ):
        super().__init__() # initialises parent class
        self.model_classification = "RN"
        self.structure = structure.structure
        self.lookback_dim = self.structure["lookback_dim"]
        self.trans_pos_emb = None
        self.trans = nn.ModuleList()
        self.pooling = nn.ModuleList()
        if "TNN" not in self.structure.keys():
            self.structure["TNN"] = []
        # if "CNN_pooling" not in self.structure.keys():
        #     self.structure["CNN_pooling"] = [[1]*len(self.structure["CNN"][0])]*len(self.structure["CNN"])
        for TNN_layer in self.structure["TNN"]:
            try:
                self.trans.append(nn.TransformerEncoderLayer(**TNN_layer,
                                                             batch_first = True).double())
            except TypeError:
                print("\nClass CNN_v1: One or more items in layer are unknown to pytorch Conv2d:\n")
                print(TNN_layer)
                raise
        self.conv = nn.ModuleList()
        self.norm = self.structure["FFNN_norm"](self.structure["FFNN"][0]["in_features"]).double()
        for pooling,CNN_layer in zip(self.structure["CNN_pooling"],self.structure["CNN"]):
            try:
                self.conv.append(nn.Conv2d(**CNN_layer).double())
            except TypeError:
                print("\nClass CNN_TNN: One or more items in layer are unknown to pytorch Conv2d:\n")
                print(CNN_layer)
                raise
            if len(CNN_layer["kernel_size"])==2:
                self.pooling.append(nn.MaxPool2d(tuple(pooling)))
            elif len(CNN_layer["kernel_size"])==3:
                self.pooling.append(nn.MaxPool3d(tuple(pooling)))
        self.FFNN_sec = self.structure["FFNN_sec"]
        self.CNN_sec = self.structure["CNN_sec"]
        self.lin = nn.ModuleList()
        self.dout = nn.ModuleList()
        self.CNN_dout = nn.ModuleList()
        self.verbose = verbose
        self.check = True
        for layer in self.FFNN_sec:
            self.dout.append(nn.Dropout(layer["dropout"]))
        for layer in self.CNN_sec:
            self.CNN_dout.append(nn.Dropout(layer["dropout"]))
        for FFNN_layer in self.structure["FFNN"]:
            self.lin.append(nn.Linear(**FFNN_layer).double())
    def forward(self, x, objective = None,**kwargs):
        # import pdb; pdb.set_trace();
        # CNN part
        if len(x.shape)>2:
            x = x[...,:self.lookback_dim]
        if len(x.shape) == 3:
            x = x.view(-1,1,self.structure["input_dim"][0],self.lookback_dim)
        for layer in range(len(self.conv)):
            x = self.CNN_dout[layer](x)
            x = self.CNN_sec[layer]["activation"](self.conv[layer](x))
            x = self.pooling[layer](x)
            if self.structure["reshape"][layer]:
                x = x.view(x.shape[0],1,*self.structure["CNN_out_dim"][layer])
        x = x.view(x.shape[0],x.shape[3],x.shape[2]*x.shape[1]).clone() #clone necessary
        # x = x.reshape(x.shape[0],x.shape[3],x.shape[2]*x.shape[1])
        # TNN part
        if self.trans_pos_emb is None:
            self.trans_pos_emb = TNN_pos_emb(x.shape[2],x.shape[1],self.structure["wavelength"])
        pos_emb = self.trans_pos_emb.get(x.shape[0]) #positional embedding
        for TNN_layer in self.trans:
            x += pos_emb
            x = TNN_layer(x)    
        del pos_emb
        if self.verbose>0 and self.check:
            shape012[1] = to.isnan(x).sum()
            self.check = False
        # x = x.reshape(*s_x)
        x = to.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.norm(x)
        for layer,secs in zip(range(len(self.lin)-1),self.FFNN_sec):
            x = self.dout[layer](x)
            x = self.lin[layer](x)
            x = secs["activation"](x)
        x = self.lin[-1](x)
        if objective is not None:
            x = objective.loss_fn.norm_pred(x)
        return x

def parameter_calculator(structure):
    # structure = TNN_s1
    struc = structure.structure
    del structure
    parameters = {}
    
    dim = struc["input_dim"]
    if "TNN" in struc.keys():
        iterator = 0
        for layer in struc["TNN"]:
            parameters["TNN"+str(iterator)] = \
                np.prod(dim)*(3 + struc["TNN"][iterator]["dim_feedforward"])
            iterator+=1
    if "CNN" in struc.keys():
        iterator = 0
        for layer in struc["CNN"]:
            subpar = []
            newdim = []
            layer = struc["CNN"][iterator]
            for subdim in range(len(dim)):
                subpar.append(layer["kernel_size"][subdim])
                if layer["dilation"][subdim] != 1:
                    subpar[subdim] = (subpar[subdim]+1)/layer["dilation"][subdim]
                newdim.append((dim[subdim]+2*layer["padding"][subdim]-layer["kernel_size"][subdim])/\
                              layer["stride"][subdim]+1)
            parameters["CNN"+str(iterator)] = np.prod(subpar)*layer["out_channels"]
            iterator+=1
        dim = newdim
    if "CNN" in struc.keys():
        dim = np.prod(dim)*layer["out_channels"]
    else:
        dim = np.prod(dim)
    if "FFNN" in struc.keys():
        iterator = 0
        for layer in struc["FFNN"]:
            parameters["FFNN"+str(iterator)] = layer["in_features"]*layer["out_features"]
            iterator+=1
    print("WARNING-result incorrect.")
    return parameters

##################################################
############  Multi-Frequency-Modules  ###########
##################################################

## example structure
multi_frequency_example_structure = {
    "kernel_order": ["w","m","q"],
    "kernels":
        {"w":{"input_dim":(41,6),"in_channels":1, "CNN":[
              {"in_channels":1, "out_channels":4, "kernel_size":(1,6), 
               "stride":(1,1), "padding": (0,0),"dilation":1},
              {"in_channels":1, "out_channels":24, "kernel_size":(41,1), 
               "stride":(1,1), "padding": (0,0),"dilation":1}],
              "CNN_sec": [{"activation":nn.Softsign()}, {"activation":nn.Softsign()}],
              "FFNN":[{"out_features":50}, {"out_features":10}],
              "CNN_norm":BatchNorm_self, "FFNN_norm":BatchNorm_self,
              "FFNN_sec": [{"activation":nn.Softsign(), "dropout":0.02},
                           {"activation":nn.Softsign(), "dropout":0.02},],
              "TNN":[{"d_model":24, "nhead":6, "dim_feedforward": 32,
               "dropout":0.2, "activation":nn.Softsign()}],
              "reshape":[True,True], "postprocessor":None, 
              },
         "m":{"input_dim":(26,3),"in_channels":1, "CNN":[
               {"in_channels":1, "out_channels":3, "kernel_size":(1,3), 
                "stride":(1,1), "padding": (0,0),"dilation":1},
               {"in_channels":1, "out_channels":16, "kernel_size":(26,1), 
                "stride":(1,1), "padding": (0,0),"dilation":1}],
               "CNN_sec": [{"activation":nn.Softsign()}, {"activation":nn.Softsign()}],
               "FFNN":[{"out_features":30}, {"out_features":10}],
               "CNN_norm":BatchNorm_self, "FFNN_norm":BatchNorm_self,
               "FFNN_sec": [{"activation":nn.Softsign(), "dropout":0.02},
                            {"activation":nn.Softsign(), "dropout":0.02},],
               "TNN":[{"d_model":16, "nhead":4, "dim_feedforward": 32,
                "dropout":0.2, "activation":nn.Softsign()}],
               "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay},
               },
         "q":{"input_dim":(76,4),"in_channels":1, "CNN":[
               {"in_channels":1, "out_channels":3, "kernel_size":(1,4), 
                "stride":(1,1), "padding": (0,0),"dilation":1},
               {"in_channels":1, "out_channels":32, "kernel_size":(76,1), 
                "stride":(1,1), "padding": (0,0),"dilation":1}],
               "CNN_sec": [{"activation":nn.Softsign()}, {"activation":nn.Softsign()}],
               "FFNN":[{"out_features":50}, {"out_features":10}],
               "CNN_norm":BatchNorm_self, "FFNN_norm":BatchNorm_self,
               "FFNN_sec": [{"activation":nn.Softsign(), "dropout":0.02},
                            {"activation":nn.Softsign(), "dropout":0.02},],
               "TNN":[{"d_model":32, "nhead":8, "dim_feedforward": 32,
                "dropout":0.2, "activation":nn.Softsign()}],
               "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay},
               },
         },
    
    "FFNN":[{"out_features":20}, {"out_features":10}, {"out_features":1},
            ],
    "FFNN_sec":[{"activation":nn.Softsign(), "dropout":0.02},
                {"activation":nn.Softsign(), "dropout":0.02},],
    }
    
class Layers_multifreq(Layers):
    def __init__(self,structure,verbose=0):
        super().__init__(structure,verbose)
        self.complete_structure(structure)
    def complete_structure(self,structure):
        # import pdb; pdb.set_trace();
        _ErrorMessage_ = "Class Layers_multifreq.__init__:"
        if "kernels" not in self.structure.keys():
            raise ValueError(_ErrorMessage_+"no kernels found")
        if "kernel_order" not in self.structure.keys():
            self.structure["kernel_order"] = list(self.structure["kernels"].keys())
        kernel_output_dim = 0
        for kernel in self.structure["kernel_order"]:
            # kernel = "q"
            self.structure["kernels"][kernel] =\
                Layers_FFNN_CNN(self.structure["kernels"][kernel]).structure
            self.structure["kernels"][kernel] =\
                Layers_TNN_CNN_FFNN(self.structure["kernels"][kernel]).structure
            if len(self.structure["kernels"][kernel]["FFNN"])>0:
                kernel_output_dim += self.structure["kernels"][kernel]["FFNN"][-1]["out_features"]
            else:
                kernel_output_dim += self.structure["kernels"][kernel]["input_dim"][0]
        
        out_features = kernel_output_dim+0
        for layer_index in range(len(self.structure["FFNN"])):
            in_features = copy.deepcopy(int(out_features))
            if self.verbose>0:
                print("""FFNN Layer {layer:2d}: receives {infeat:6d} input_features""".\
                format(layer = layer_index,
                       infeat = in_features))
            layer = self.structure["FFNN"][layer_index]
            layer["in_features"] = copy.deepcopy(in_features)
            self.structure["FFNN"][layer_index] = copy.deepcopy(layer)
            out_features = int(layer["out_features"])
        
    def serialise(self):
        pass
    @classmethod
    def deserialise(cls):
        pass
    @classmethod
    def from_dict(cls):
        pass
    
multi_frequency_example_structure = Layers_multifreq(multi_frequency_example_structure)

class ANN_MultiFreq(nn.Module):
    '''
    Class to handle multi frequency input.
    '''
    def __init__(self,
                 structure:Layers_multifreq, ### do we need another new structure?
                 verbose = 0,
                 ):
        super().__init__() # initialises parent class
        self.model_classification = "RN"
        self.structure = structure.structure
        self.lookback_dim = {}; self.trans_pos_emb = {}
        self.check = {}
        
        if "seed" in self.structure.keys():
            to.manual_seed(self.structure["seed"])
        else:
            to.manual_seed(np.random.randint(0,10000))
        
        ## initialise module containers for all parameter layers
        self.kernels_conv = nn.ModuleList(); self.kernels_conv_pos = {}; conv_counter = 0;
        self.kernels_pooling = nn.ModuleList(); self.kernels_pooling_pos = {}; pooling_counter = 0;
        self.kernels_trans = nn.ModuleList(); self.kernels_trans_pos = {}; trans_counter = 0;
        self.kernels_lin = nn.ModuleList(); self.kernels_lin_pos = {}; lin_counter = 0;
        self.kernels_dout = nn.ModuleList(); self.kernels_dout_pos = {}; dout_counter = 0;
        self.kernels_CNN_dout = nn.ModuleList(); self.kernels_CNN_dout_pos = {}; CNN_dout_counter = 0;
        self.kernels_norm = nn.ModuleList(); self.kernels_norm_pos = {}; norm_counter = 0;
        self.kernels_FFNN_sec = {}; self.kernels_CNN_sec = {};
        self.kernels_postprocessor = nn.ModuleList(); self.kernels_postprocessor_pos = {}; postprocessor_counter = 0;
        for frequency in self.structure["kernel_order"]:
            self.lookback_dim[frequency] = self.structure["kernels"][frequency]["lookback_dim"]
            self.trans_pos_emb[frequency] = None
            self.kernels_conv_pos[frequency] = []; self.kernels_pooling_pos[frequency] = [];
            self.kernels_trans_pos[frequency] = []; self.kernels_lin_pos[frequency] = [];
            self.kernels_dout_pos[frequency] = []; self.kernels_CNN_dout_pos[frequency] = [];
            
            if "TNN" not in self.structure["kernels"][frequency].keys():
                self.structure["kernels"][frequency]["TNN"] = []
            for TNN_layer in self.structure["kernels"][frequency]["TNN"]:
                try:
                    self.kernels_trans.append(nn.TransformerEncoderLayer(
                        **TNN_layer, batch_first = True).double())
                except TypeError:
                    print("\nClass ANN_MultiFreq.__init__: One or more items in layer are unknown to pytorch Conv2d:\n")
                    print(TNN_layer)
                    raise
                self.kernels_trans_pos[frequency].append(trans_counter+0)
                trans_counter += 1
            self.kernels_norm.append(self.structure["kernels"][frequency]["FFNN_norm"](
                self.structure["kernels"][frequency]["input_dim"][0]).double())
            self.kernels_norm_pos[frequency] = norm_counter+0
            norm_counter+=1
            
            ## convolutional layers
            self.kernels_conv_pos[frequency] = []; self.kernels_pooling_pos[frequency] = []
            for pooling,CNN_layer in zip(
                    self.structure["kernels"][frequency]["CNN_pooling"],
                    self.structure["kernels"][frequency]["CNN"]):
                try:
                    self.kernels_conv.append(nn.Conv2d(**CNN_layer).double())
                    self.kernels_conv_pos[frequency].append(conv_counter+0)
                    conv_counter+=1
                except TypeError:
                    print("\nClass CNN_TNN: One or more items in layer are unknown to pytorch Conv2d:\n")
                    print(CNN_layer)
                    raise
                if len(CNN_layer["kernel_size"])==2:
                    self.kernels_pooling.append(nn.MaxPool2d(tuple(pooling)))
                elif len(CNN_layer["kernel_size"])==3:
                    self.kernels_pooling.append(nn.MaxPool3d(tuple(pooling)))
                self.kernels_pooling_pos[frequency].append(pooling_counter)
                pooling_counter+=1
            self.kernels_FFNN_sec[frequency] = self.structure["kernels"][frequency]["FFNN_sec"]
            self.kernels_CNN_sec[frequency] = self.structure["kernels"][frequency]["CNN_sec"]
            self.kernels_lin_pos[frequency] = []
            self.kernels_dout_pos[frequency] = []
            self.kernels_CNN_dout_pos[frequency] = []
            for layer in self.kernels_FFNN_sec[frequency]:
                self.kernels_dout.append(nn.Dropout(layer["dropout"]))
                self.kernels_dout_pos[frequency].append(dout_counter+0)
                dout_counter+=1
            for layer in self.kernels_CNN_sec[frequency]:
                self.kernels_CNN_dout.append(nn.Dropout(layer["dropout"]))
                self.kernels_CNN_dout_pos[frequency].append(CNN_dout_counter+0)
                CNN_dout_counter+=1
            for FFNN_layer in self.structure["kernels"][frequency]["FFNN"]:
                self.kernels_lin.append(nn.Linear(**FFNN_layer).double())
                self.kernels_lin_pos[frequency].append(lin_counter+0)
                lin_counter +=1
            if self.structure["kernels"][frequency]["postprocessor"] is not None:
                if len( self.structure["kernels"][frequency]["FFNN"]) >0:
                    self.kernels_postprocessor.append(
                        self.structure["kernels"][frequency]["postprocessor"]["layer_type"](
                            self.structure["kernels"][frequency]["FFNN"][-1]["out_features"]))
                else:
                    self.kernels_postprocessor.append(
                        self.structure["kernels"][frequency]["postprocessor"]["layer_type"](
                            self.structure["kernels"][frequency]["input_dim"][0]))
                self.kernels_postprocessor_pos[frequency] = postprocessor_counter+0
                postprocessor_counter+=1
            
            self.check[frequency] = False
        
        self.FFNN_sec = self.structure["FFNN_sec"]
        self.lin = nn.ModuleList()
        self.dout = nn.ModuleList()
        self.verbose = verbose
        for layer in self.FFNN_sec:
            self.dout.append(nn.Dropout(layer["dropout"]))
        for FFNN_layer in self.structure["FFNN"]:
            self.lin.append(nn.Linear(**FFNN_layer).double())
    
    def forward(self, x:dict,meta:dict={}, objective = None):
        # verbose = False
        # import pdb; pdb.set_trace(); verbose = True
        # CNN part
        for f in self.structure["kernel_order"]:
            # f = "q"
            if len(x[f].shape)>2:
                x[f] = x[f][...,:self.lookback_dim[f]]
            if len(x[f].shape) == 3:
                x[f] = x[f].view(-1,1,self.structure["kernels"][f]["input_dim"][0],
                                 self.lookback_dim[f])
            if len(self.kernels_conv_pos[f])>0:
                layer_0 = self.kernels_conv_pos[f][0]
                for layer in self.kernels_conv_pos[f]:
                    # layer = self.kernels_conv_pos[f][1]
                    x[f] = self.kernels_CNN_dout[layer](x[f])
                    x[f] = self.kernels_CNN_sec[f][layer-layer_0]["activation"](self.kernels_conv[layer](x[f]))
                    x[f] = self.kernels_pooling[layer](x[f])
                    if self.structure["kernels"][f]["reshape"][layer-layer_0]:
                        x[f] = x[f].view(x[f].shape[0],1, \
                            *self.structure["kernels"][f]["CNN_out_dim"][layer-layer_0])
            x[f] = x[f].view(x[f].shape[0],x[f].shape[3],x[f].shape[2]*x[f].shape[1]).clone() #clone necessary
            # x = x.reshape(x.shape[0],x.shape[3],x.shape[2]*x.shape[1])
            # TNN part
            if len(self.kernels_trans_pos[f])>0:
                if self.trans_pos_emb[f] is None:
                    self.trans_pos_emb[f] = TNN_pos_emb(
                        x[f].shape[2],x[f].shape[1], self.structure["kernels"][f]["wavelength"])
                pos_emb = self.trans_pos_emb[f].get(x[f].shape[0]) #positional embedding
                for TNN_layer_index in self.kernels_trans_pos[f]:
                    x[f] += pos_emb
                    x[f] = self.kernels_trans[TNN_layer_index](x[f])    
                del pos_emb
            if self.verbose>0 and self.check[f]:
                shape012[1] = to.isnan(x[f]).sum()
                self.check[f] = False
            # x = x.reshape(*s_x)
            x[f] = to.flatten(x[f], 1) # flatten all dimensions except the batch dimension
            x[f] = self.kernels_norm[self.kernels_norm_pos[f]](x[f])
            for layer,secs in zip(self.kernels_lin_pos[f],self.kernels_FFNN_sec[f]):
                # layer = self.kernels_lin_pos[f][1]; secs = self.kernels_FFNN_sec[f][1]
                x[f] = self.kernels_dout[layer](x[f])
                x[f] = self.kernels_lin[layer](x[f])
                x[f] = secs["activation"](x[f])
            # x[f] = self.lin[self.kernels_lin_pos[f][-1]](x[f])
        
        for f in meta.keys():
            # f = "m"
            x[f] = self.kernels_postprocessor[self.kernels_postprocessor_pos[f]](x[f],meta[f])
                
        y = to.cat(tuple(x.values()),dim=1)
        del x
        
        for layer,secs in zip(range(len(self.lin)-1),self.FFNN_sec):
            y = self.dout[layer](y)
            y = self.lin[layer](y)
            y = secs["activation"](y)
        y = self.lin[-1](y)
        
        # import pdb; pdb.set_trace();
        
        if objective is not None:
            y = objective.loss_fn.norm_pred(y)
                
        return y


################################################################################
################################################################################
######################  Dataset Class Preprocessing Units  #####################
################################################################################
################################################################################

def bpp_normalise(x:np.ndarray,dim:int=0):
    mean_values = x.mean(axis=dim)
    std_values = x.std(axis=dim)
    x=(x-mean_values)/std_values
    return x


def bpp_rank_min_max(x:np.ndarray,dim:int=0):
    x = bpp_rank(x,dim)
    x = bpp_min_max(x,dim)
    return x

def bpp_rank(x:np.ndarray,dim:int=0):
    '''
    Converts dimension-wise [dim] into ranked data.
    
    Parameters
    ----------
    x : np.ndarray
        Input matrix.
    dim : int DEFAULT 1
        Dimension of ranks. Default is 0, which makes data sorted columnwise

    Returns
    -------
    x : np.ndarray
        Ranked matrix.

    Testing:
        x = np.array([[1,2,5,4],
                      [0.5,0.1,3.7,5],
                      [-0.6,0.3,7,15],
                      [4,0.1,6,13],
                      [-1,2,5.1,16]])
        dim = 0
        x = pp_rank(x,dim=0)
    '''
    x = rankdata(x,method = "average",axis=dim)
    return x

def bpp_min_max(x:np.ndarray,dim:int=0,span = 3):
    '''
    Rearranges data into range between +- 3

    Parameters
    ----------
    x : np.ndarray
        Input matrix.
    dim : int DEFAULT 1
        Dimension of scaling. Default is 0, which makes data scaled columnwise

    Returns
    -------
    x: np.ndarray
        Minimum-maximum scaled data

    Testing:
        x = np.array([[2, 2, 1, 0],
                      [1, 0, 0, 1],
                      [0, 1, 2, 2]])
        dim = 0
        span = 3
        pp_min_max(x,dim=0)
    '''
    max_values = x.max(axis=dim)
    min_values = x.min(axis=dim)
    mean_values = (max_values+min_values)/2
    mean_distances = max_values-mean_values
    x=(x-mean_values)/mean_distances*span
    return x

class PPB_nan_drop_df():
    def __init__(self,**kwargs):
        pass
    def __call__(self,x:pd.DataFrame):
        index_mask = (x.isna().sum(axis=1)==0)
        x = x.loc[index_mask,:]
        return x

class PPB_nan_mean_df():
    def __init__(self,rolling = 0,date_format = "%Y-%m-%d", function = "mean", **kwargs):
        self.rolling = rolling
        self.date_format = date_format
        self.function = function
    def __call__(self,x:pd.DataFrame):
        # import pdb; pdb.set_trace();
        rolling = self.rolling
        date_format = self.date_format
        function = self.function
        # x = pd.read_hdf("D:/BT/test_data/test_m_q1/data_b0xq.npy","data")
        date_entity = list(set(x.index.names).intersection(set(PhD4_dataset.date_entities)))[0]
        x_columns = x.columns
        x.sort_values(date_entity,inplace=True)
        if rolling == 0:
            for column in x_columns:
                if function == "mean":
                    col_mean= x[column].groupby(level=date_entity).mean().to_frame("__col__")
                elif function == "median":
                    col_mean= x[column].groupby(level=date_entity).median().to_frame("__col__")
                col = pd.merge(x[column],col_mean,left_index=True, right_index = True,how = "left")
                col.loc[col[column].isna(),column] = col.loc[col[column].isna(),"__col__"]
                x[column] = col[column]
        else:
            if function == "mean":
                func_apl = np.nanmean
            elif function == "median": # WARNING: performance much worse
                func_apl = np.nanmedian
            # timer = timeit()
            date_0 = dt.datetime.today()
            x["date_int"] = x.index.get_level_values(date_entity)
            if type(x["date_int"].values[0]) is str:
                x["date_int"] = x["date_int"].apply(lambda x: dt.datetime.strptime(x,date_format))
            x["date_int"] = (x["date_int"]-date_0).dt.days
            x["date_int"]-=x["date_int"].values[0]
            dates = x["date_int"].to_numpy()
            del x["date_int"]
            
            x_index = x.index
            x_values = x.to_numpy()
            del x
            x_mean_corrected = np.ndarray(shape = x_values.shape)
            max_index,min_index = -1, 0
            while max_index+1 != len(dates):
                index_finder = np.where(dates==dates[max_index+1])[0]
                max_index = index_finder[-1]
                int_index = index_finder[0]
                min_index = np.where(dates>=dates[max_index]-rolling)[0][0]
                x_mean_corrected[int_index:max_index+1,:] = \
                    x_values[int_index:max_index+1,:]
                x_mean_corrected[int_index:max_index+1,:] = \
                    np.where(np.isnan(x_mean_corrected[int_index:max_index+1,:]),
                             np.nan_to_num(func_apl(x_values[min_index:max_index+1,:],
                                                    axis=0)),
                             x_mean_corrected[int_index:max_index+1,:])
            x = pd.DataFrame(x_mean_corrected, index = x_index,columns = x_columns)
        return x.replace(np.nan,0)

class PPB_nan_to_0():
    def __init__(*args,**kwargs):
        pass
    def __call__(self,x,*args,**kwargs):
        # import pdb; pdb.set_trace();
        return x.replace(np.nan,0)

class PPB_norm_df():
    def __init__(self,rolling = 0,date_format = "%Y-%m-%d", span = 1.96, **kwargs):
        self.rolling = rolling
        self.date_format = date_format
        self.span = span
    def __call__(self,x:pd.DataFrame):
        rolling = self.rolling
        date_format = self.date_format
        span = self.span
        # import pdb; pdb.set_trace();
        # x = pd.read_hdf("D:/BT/test_data/test_m_q1/data_b0xq.npy","data")
        # x = pp_nan_df(x,rolling,date_format,function)
        date_entity = list(set(x.index.names).intersection(set(PhD4_dataset.date_entities)))[0]
        x_columns = x.columns
        x.sort_values(date_entity,inplace=True)
        if rolling == 0:
            ## reasons for iteration: dataset may be very large. In that case, it is not ideal to
            ## copy the dataset twice over, so I decided to make the normalisation per column.
            for column in x_columns:
                col_df = pd.DataFrame()
                col_df["__mean__"] = x[column].groupby(level=date_entity).mean().to_frame("__mean__")
                col_df["__mean__"] = col_df["__mean__"].replace(np.nan,0)
                col_df["__std__"] = x[column].groupby(level=date_entity).std().to_frame("__std__")
                col_df["__std__"] = col_df["__std__"].replace([0,np.nan],1)
                col = pd.merge(x[column],col_df,left_index=True, right_index = True,how = "left")
                col[column] = (col[column]-col["__mean__"])/col["__std__"]
                x[column] = col[column]
        else:
            #  rolling = 91
            date_0 = dt.datetime.today()
            x["date_int"] = x.index.get_level_values(date_entity)
            date_type = type(x["date_int"].values[0])
            if date_type is str:
                x["date_int"] = x["date_int"].apply(lambda x: dt.datetime.strptime(x,date_format))
            elif date_type is not np.datetime64:
                x["date_int"] = x["date_int"].astype("datetime64[ns]")
            x["date_int"] = (x["date_int"]-date_0).dt.days
            x["date_int"]-=x["date_int"].values[0]
            dates = x["date_int"].to_numpy()
            del x["date_int"]
            
            x_index = x.index
            x_values = x.to_numpy()
            del x
            x_norm_processed = np.ndarray(shape = x_values.shape)
            max_index,min_index = -1, 0
            while max_index+1 != len(dates):
                # 
                index_finder = np.where(dates==dates[max_index+1])[0]
                max_index = index_finder[-1]
                int_index = index_finder[0]
                min_index = np.where(dates>=dates[max_index]-rolling)[0][0]
                x_norm_processed[int_index:max_index+1,:] = \
                    x_values[int_index:max_index+1,:]
                
                x_mean = np.nanmean(x_values[min_index:max_index+1,:],axis=0)
                x_std = np.nanstd(x_values[min_index:max_index+1,:],axis=0)
                x_std[x_std == 0] = 1
                x_std*=(span/1.96)
                
                x_norm_processed[int_index:max_index+1,:] = \
                    (x_norm_processed[int_index:max_index+1,:]-x_mean)/\
                        x_std
            x = pd.DataFrame(x_norm_processed, index = x_index,columns = x_columns)
        return x.replace(np.nan,0)

class PPB_rank_df():
    def __init__(self,rolling = 0,date_format = "%Y-%m-%d", span = 3, **kwargs):
        self.rolling = rolling
        self.date_format = date_format
        self.span = span        
    def __call__(self,x:pd.DataFrame):
        rolling = self.rolling
        date_format = self.date_format
        span = self.span
        # import pdb; pdb.set_trace();
        # x = pd.read_hdf("D:/BT/test_data/test_m_q1/data_b0xq.npy","data")
        # x = pp_nan_df(x,rolling,date_format,function)
        date_entity = list(set(x.index.names).intersection(set(PhD4_dataset.date_entities)))[0]
        x_columns = x.columns
        x.sort_values(date_entity,inplace=True)
        if rolling == 0:
            x = pd.merge(x,x.groupby(date_entity).size().to_frame("size")/2,
                         left_index=True, right_index=True,how="inner")
            for column in x_columns:
                x[column] = (x[column].groupby(date_entity).rank(method="average")-x["size"])/x["size"]*span
            del x["size"]
        else:
            #  rolling = 93
            date_0 = dt.datetime.today()
            x["date_int"] = x.index.get_level_values(date_entity)
            date_type = type(x["date_int"].values[0])
            if date_type is str:
                x["date_int"] = x["date_int"].apply(lambda x: dt.datetime.strptime(x,date_format))
            elif date_type is not np.datetime64:
                x["date_int"] = x["date_int"].astype("datetime64[ns]")
            x["date_int"] = (x["date_int"]-date_0).dt.days
            x["date_int"]-=x["date_int"].values[0]
            dates = x["date_int"].to_numpy()
            del x["date_int"]
            
            x_index = x.index
            x_values = x.to_numpy()
            del x
            x_norm_processed = np.ndarray(shape = x_values.shape)
            max_index,min_index = -1, 0
            while max_index+1 != len(dates):
                index_finder = np.where(dates==dates[max_index+1])[0]
                max_index = index_finder[-1]
                int_index = index_finder[0]
                min_index = np.where(dates>=dates[max_index]-rolling)[0][0]
                local_shape_half = (max_index-min_index)/2
                local_array = x_values[min_index:max_index+1,:]
                local_array_sorted = local_array.argsort(axis=0).argsort(axis=0).astype(float)
                local_na = 2*local_shape_half-np.isnan(local_array).sum(axis=0) #
                local_na[local_na==0]=np.nan
                
                masked_local = np.ma.masked_array(
                    data=local_array_sorted,
                    mask=(local_array_sorted>local_na))
                
                #### nan value handling
                ## reset the lowest x values per column to np.nan.
                del local_array_sorted
                local_array_sorted = masked_local.filled(np.nan)
                del masked_local
                local_array_sorted/=local_na
                del local_na
                
                local_array_sorted = local_array_sorted[int_index-min_index:,:]
                local_array_sorted-=.5
                local_array_sorted*=2*span
                
                x_norm_processed[int_index:max_index+1,:] = local_array_sorted
            x = pd.DataFrame(x_norm_processed, index = x_index,columns = x_columns)
        return x.replace(np.nan,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

################################################################################
################################################################################
####################  Loss Functions, Penalties, Objectives  ###################
################################################################################
################################################################################

##################################################
################# Loss Functions #################
##################################################

class Loss_MSE_weighted():
    def __init__(self,
                 **kwargs):
        pass
    def __call__(self,output,target,model=None,side=[], **kwargs):
        # import pdb; pdb.set_trace()
        if len(side) == 0:
            loss = ((output-target)**2).sum()/len(output)
        else:
            loss = (((output-target)**2*side)).sum()/sum(side)#*len(output)
        return loss

class Loss_Huber_weighted():
    def __init__(self,threshold=0.01,**kwargs):
        self.threshold = threshold
    def __call__(self,output,target,model=None,side=[],**kwargs):
        # import pdb; pdb.set_trace()
        residuals_raw = to.abs(output-target)
        # residuals = to.clip(output-target,-self.threshold,self.threshold)
        residuals = to.where(residuals_raw<self.threshold,
                             residuals_raw**2,self.threshold*(residuals_raw*2-self.threshold))
        
        if len(side) == 0:
            loss = (residuals).sum()/len(output)
        else:
            loss = (residuals*side).sum()/sum(side)#*len(output)
        return loss

class Loss_QLikelihood():
    def __init__(self,cap:float = .1**8,mode = "cap",**kwargs):
        '''
        This class represents the QLikelihood ratio:
            QLikelihood ratio := test/pred - np.log(test/pred) - 1

        Parameters
        ----------
        cap : float DEFAULT 0.1**8
            Cap variable used for dealing with zero values violating the nature of the test.
        mode : str DEFAULT "cap"
            Mode for dealing with zero values violating the test capabilities.
            Must be one of:
                "cap": uses the cap variable to impose a lower cap.
                "drop": drops observations lower than cap.
                "add": adds cap to all values, imposing a slight bias into the ratio.
        **kwargs : optional arguments for scaled use. Not use.
        '''
        self.cap = cap; 
        if mode not in ["cap","drop","add"]:
            raise ValueError("Class Loss_QLikelihood received illegal value for parameter mode:\n"+\
                             f'Expected: one of ["cap","drop","add"]\nGot:{mode}')
        self.mode = mode;
    def __call__(self, output, target,**kwargs):
        ## ensure that the series are not overwritten by this class:
        output = output.copy()
        target = target.copy()
        
        ## treatment of zero values
        if self.mode == "cap":
            nonzero = (target<=self.cap)
            if nonzero.sum() == np.prod(target.shape[0]):
                raise ValueError("Class Loss_QLikelihood.__call__:All entries in target are zero.")
            else:
                target[nonzero] = self.cap
            
            nonzero = (output<=self.cap)
            if nonzero.sum() == np.prod(target.shape[0]):
                raise ValueError("Class Loss_QLikelihood.__call__:All entries in output are zero.")
            else:
                output[nonzero] = self.cap
            del nonzero
            
        elif self.mode == "drop":
            mask = (output>=self.cap) & (target>=self.cap)
            if mask.sum() == 0:
                raise ValueError("All entries are zero")
            output = output[mask]; target = target[mask]
            del mask
        elif self.mode == "add":
            output+=self.cap
            target+=self.cap
        
        loss = (target/output-np.log(target/output)-1).mean()
        return loss

class Loss_HML(nn.Module):
    def __init__(
            self,
            return_mode = "log",
            penalties = [],
            mode = "ret", # one of ["ret","SR"]
            norm_mode = "HML_flex", # one of ["HML","HML_flex"]
            HML_par = .9, ## ratio of HML for "HML", threshold mid to end for "HML_flex"
            norm_function = nn.Softsign(),
            **kwargs):
        
        super(Loss_HML, self).__init__()
        norm_min = norm_function(to.tensor([-100000])); norm_max = norm_function(to.tensor([100000])); 
        norm_0 = norm_function(to.tensor([0]));
        if abs((norm_max-norm_0)-(norm_0-norm_min))/2>10**-8:
            raise ValueError("Class: Loss_HML: __init__: norm function must be symetric around 0 and"+\
                             " continuously defined in |R.")
        elif HML_par <= 0:
            raise ValueError("Class: Loss_HML: __init__: HML_par must be >0!")
        elif mode not in ["ret","SR"]:
            raise ValueError('Class: Loss_HML: __init__: mode must be one of ["ret","SR"]!')
        elif norm_mode not in ["HML","HML_flex"]:
            raise ValueError('Class: Loss_HML: __init__: norm_mode must be one of ["HML","HML_flex"]!')
        self.lower = to.tensor([norm_min*(HML_par)+norm_0*(1-HML_par)])
        self.upper = to.tensor([norm_max*HML_par+norm_0*(1-HML_par)])
        self.return_mode = return_mode
        self.penalties = penalties
        self.mode = mode
        self.norm_mode = norm_mode
        self.HML_par = HML_par
        self.norm_function = norm_function
    
    def forward(self,output,target,side=None,apply = False,model=None,**kwargs):
        ## apply penalties
        # import pdb; pdb.set_trace()
        penalty = 0
        for penalty_exec in self.penalties:
            penalty += penalty_exec(output,side)
            
        ## standardise returns to simple
        if self.return_mode == "log":
            target = target.exp()-1
            
        if self.norm_mode == "HML":
            low_threshold = output.quantile(self.HML_par/2)
            high_threshold = output.quantile(1-self.HML_par/2)
            indices_low = output<low_threshold
            indices_high = output>high_threshold
            
        elif self.norm_mode == "HML_flex":
            indices_low = output<self.lower.to(output.device)
            indices_high = output>self.upper.to(output.device)
            
            
        transformed_output = to.where(indices_low, to.tensor(
            -1.0, dtype=output.dtype, device=output.device), output)
        transformed_output = to.where(indices_high, to.tensor(
            1.0, dtype=output.dtype, device=output.device), transformed_output)
        transformed_output = to.where(~(indices_low|indices_high), 
                                      to.tensor(0.0, dtype=output.dtype, device=output.device), 
                                      transformed_output)
        
        
        # transformed_output = to.zeros_like(output)
        # transformed_output[~(indices_low | indices_high)] = 0
        # transformed_output[indices_low] = -1
        # transformed_output[indices_high] = 1
            
        transformed_output = transformed_output/max(1,abs(transformed_output).sum())
        
        ## calculate loss
        if self.mode == "ret":
            loss = 1-(transformed_output*target).sum()
        elif self.mode == "SR":
            losses = []
            for bootstrap in range(10):
                portfolios = []
                indices = np.arange(len(transformed_output))
                np.random.shuffle(indices)
                list_of_indices = np.array_split(indices,10)
                for indices_index in range(10):
                    # indices_index=0
                    local_index = list_of_indices[indices_index]
                    portfolios.append((transformed_output[local_index]*target[local_index]).sum()/
                                      max(1,abs(transformed_output[local_index]).sum()))
                portfolios = to.stack(portfolios)
                mean = to.mean(portfolios); std = to.std(portfolios);
                if mean==0 and std==0:
                    losses.append(mean)
                elif std==0:
                    losses.append(mean/.0001)
                else:
                    losses.append(mean/std)
            losses = to.stack(losses)
            loss = -losses.mean()
        return loss, penalty
    
    def norm_pred(self,output):
        # import pdb; pdb.set_trace()
        output = self.norm_function(output)
            
        # if self.norm_mode == "HML":
        #     low_threshold = output.quantile(self.HML_par/2)
        #     high_threshold = output.quantile(1-self.HML_par/2)
        #     indices_low = output<low_threshold
        #     indices_high = output>high_threshold
            
        # elif self.norm_mode == "HML_flex":
        #     indices_low = output<self.lower.to(output.device)
        #     indices_high = output>self.upper.to(output.device)
            
        # output[~(indices_low | indices_high)] = 0
        # output[indices_low] = -1
        # output[indices_high] = 1
        
        # output/=max(1,abs(output).sum())
        
        return output
        

class Loss_ENET(nn.Module):
    def __init__(self,
                 base_penalty=Loss_MSE_weighted(),
                 alpha_p=0.000001,
                 lambda_p=0.00001,
                 mode = "equal", # one of ["equal","value"]
                 penalties=[],
                 volatility = False,
                 **kwargs):
        super(Loss_ENET, self).__init__()
        self.base_penalty = base_penalty
        self.alpha_p = alpha_p
        self.volatility = volatility
        self.lambda_p = lambda_p
        self.penalties = penalties
        # self.norm_pred_kernel = nn.ReLU()
    def forward(self,output,target,side=[],model=None, **kwargs):
        # import pdb; pdb.set_trace()
        loss = self.base_penalty(output,target,side)
        penalty = 0
        if model is None:
            return loss
        ridge = sum((parameter**2).sum() for parameter in model.parameters())
        lasso = sum(parameter.abs().sum() for parameter in model.parameters())
        std = 0
        # std = max(0,.01-output.std())
        penalty += self.lambda_p *(self.alpha_p*lasso+(1-self.alpha_p)*ridge)+std
        for penalty_exec in self.penalties:
            penalty += penalty_exec(output)
        return loss, penalty
    def norm_pred(self,output):
        # import pdb; pdb.set_trace()
        if self.volatility:
            pass
            # output = to.abs(output)
            output = output*output
            # output = to.exp(output)
            # output = self.norm_pred_kernel(output)
            # output = to.clip(output,min=0)
        return output

class Loss_SDF(nn.Module):
    def __init__(self,
                 normalise = None,
                 return_mode = "log",
                 penalties = [],
                 mode = "SDF", # one of ["SDF","SDF_s","SR"]
                 lambda_s = .1,
                 # limit = None
                 **kwargs):
        super(Loss_SDF, self).__init__()
        self.return_mode = return_mode
        self.normalise = normalise
        if "to" in self.normalise:
            self.min_weightsum, self.max_weightsum = [float(i) for i in self.normalise.split("to")]
            self.normalise = "to"
        self.penalties = penalties
        self.mode = mode
        self.lambda_s = lambda_s
        # self.limit = limit
    def forward(self,output,target,side=None,apply = False,model=None,**kwargs):
        # import pdb; pdb.set_trace()
        # timer = timeit(10)
        penalty = 0
        if not apply:
            for penalty_exec in self.penalties:
                penalty += penalty_exec(output,side)
        else:
            for penalty_exec in self.penalties:
                output = penalty_exec.oos(output,side)
        # print("loss_calc:",timer("penalties"))
        if self.return_mode == "log":
            target = target.exp()-1
        if self.mode == "SDF" or apply:
            loss = 1-(output*target).sum()
        elif self.mode == "SDF_s":
            loss = 1-(output*(target-self.lambda_s*side)).sum()
        elif self.mode == "SR":
            loss = -(output*(target/side)).sum()
        # print("loss_calc:",timer("loss_calc"))
        return loss, penalty
    def norm_pred(self,output):
        # import pdb; pdb.set_trace()
        if self.normalise == "1": # rescale portfolio to 100% investment
            tot_sum = output.sum()
            if tot_sum<0:
                output[output<0] /= -2*(output[output<0].sum().abs()/max(.01,output[output>0].sum()))
                tot_sum = output.sum()
            output= output/ tot_sum
        elif self.normalise == "1a": #rescale to absolute of 1
            output= output/ output.abs().sum()
        elif self.normalise == "0d": # decline larger absolute side of 0
            if -output[output<0].sum() > output[output>0].sum():
                output[output<0] = output[output<0]/(output[output>0].sum()/-output[output<0].sum())
            elif -output[output<0].sum() < output[output>0].sum():
                output[output>0] = output[output>0]/(output[output<0].sum()/-output[output>0].sum())
        elif self.normalise == "0r": # increase smaller absolute side of 0
            if -output[output<0].sum() > output[output>0].sum():
                output[output<0] = output[output<0]/(output[output>0].sum()/-output[output<0].sum())
            elif -output[output<0].sum() < output[output>0].sum():
                output[output>0] = output[output>0]/(output[output<0].sum()/-output[output>0].sum())
        elif "to" == self.normalise:
            total_sum = output.abs().sum()
            with to.no_grad():
                if total_sum>self.max_weightsum:
                    output/= (total_sum/self.max_weightsum)
                if total_sum<self.min_weightsum:
                    output*= (self.min_weightsum/total_sum)
        return output
    
##################################################
#################### Penalties ###################
##################################################

class OBJ_P():
    @classmethod
    def from_dict(cls,pars):
        if "weight_power" == pars["name"]:
            return OBJ_P_weight_power(**pars)
        elif "long_short_ratio" == pars["name"]:
            return OBJ_P_long_short_ratio(**pars)
        elif "weights_value" == pars["name"]:
            return OBJ_P_weights_value(**pars)
        elif "weights_stdmin" == pars["name"]:
            return OBJ_P_weights_stdmin(**pars)
        elif "weights_non_zero" == pars["name"]:
            return OBJ_P_weights_non_zero(**pars)
        elif "mc_max" == pars["name"]:
            return OBJ_P_mc_max(**pars)
        
class OBJ_P_weight_power(OBJ_P):
    #penalises the power of the output values over a certain tolerance
    def __init__(self,l_value=1400,power=2,tolerance = .00005,verbose=0,**kwargs):
        '''
        Ranges for parameters:
            l_value:    any value desired for strength of penalty.
            power:      should be two, could be between 1.5 and 3
            tolerance:  at power two, it should be between .000025 and .000075
        '''
        self.power=power
        self.apply = False
        self.l_value = l_value
        self.tolerance = tolerance
        self.verbose = verbose
    def __call__(self,output,side=[]):
        # import pdb; pdb.set_trace();
        output = output.abs()
        pen_value = (abs(output)**self.power).mean()*((len(output)/256))#**(self.power-.5))
        if self.verbose>0:
            return max(0,pen_value-self.tolerance)*self.l_value,pen_value*self.l_value
        return max(0,pen_value-self.tolerance)*self.l_value
    def oos(self,output,side=[]):
        return output
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["power", "l_value", "tolerance"]}
        pars["name"] = "weight_power"
        return pars
    
class OBJ_P_long_short_ratio(OBJ_P):
    # penalises long-short ratio constraint violations
    def __init__(self,l_value=.1,ls_max = 1, sl_max = 2/3,verbose=0,**kwargs):
        '''
        Ranges for parameters:
            l_value:    any value desired for strength of penalty.
            ls_max:     maximum fraction of weights to be positive, can be any value between 0 and 1,
                        lets say .8 to 1
            sl_max:     maximum fraction of weights to be negative, can be any value between 0 and 1
                        lets say .5 to .8
        '''
        self.ls_max,self.sl_max = ls_max, sl_max
        self.apply = False
        self.l_value = l_value
        self.verbose = verbose
    def __call__(self,output,side=[]):
        # import pdb; pdb.set_trace()
        neg_sum = output[output<0].sum().abs()
        pos_sum = output[output>0].sum()
        total = neg_sum+pos_sum
        neg_ratio = neg_sum/total
        pos_ratio = pos_sum/total
        del neg_sum,pos_sum
        penalty = (max(0,neg_ratio-self.sl_max)+\
                max(0,pos_ratio-self.ls_max))*\
            self.l_value
        if self.verbose>0:
            if neg_ratio>pos_ratio:
                pre_tol = -neg_ratio
            else:
                pre_tol = pos_ratio
            return penalty, pre_tol
        return penalty
    def oos(self,output,side=[]):
        return output
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["ls_max", "sl_max", "l_value"]}
        pars["name"] = "long_short_ratio"
        return pars
        
class OBJ_P_weights_value(OBJ_P):
    # penalises exceet of certain value of output values
    def __init__(self,l_value=.1,max_val = .02,verbose=0,**kwargs):
        """
        Ranges for parameters:
            l_value:    any value desired for strength of penalty.
            max_val:    threshold over which to penalise, should be between .01 and .05
        """
        self.max_val = max_val
        self.apply = False
        self.l_value = l_value
        self.verbose = verbose
    def __call__(self,output,side=[]):
        # import pdb; pdb.set_trace()
        output = output.abs()
        penalty = output[output>self.max_val].sum()*self.l_value
        if self.verbose>0:
            return penalty,penalty
        return penalty
    def oos(self,output,side=[]):
        return output
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["max_val", "l_value"]}
        pars["name"] = "weights_value"
        return pars
    
class OBJ_P_weights_stdmin(OBJ_P):
    # penalises exceet of certain value of output values
    def __init__(self,l_value=.2,min_val = .00225,verbose=0,**kwargs):
        """
        Ranges for parameters:
            l_value:    any value desired for strength of penalty.
            min_val:    threshold of minimum standard deviation, should be between .001 and .0035
        """
        self.min_val = min_val
        self.apply = False
        self.l_value = l_value/min_val
        self.verbose = verbose
    def __call__(self,output,side=[]):
        # import pdb; pdb.set_trace()
        std = output.std()*(len(output)/256)
        if self.verbose>0:
            return max(0,self.min_val-std)*self.l_value, (self.min_val-std)*self.l_value
        return max(0,self.min_val-std)*self.l_value
    def oos(self,output,side=[]):
        return output
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["min_val", "l_value"]}
        pars["name"] = "weights_stdmin"
        return pars
       
class OBJ_P_weights_non_zero(OBJ_P):
    # penalises the fraction of non-zero weights
    def __init__(self,l_value=2,non_zero_tol=.95,zero_tol=0.00001,verbose=0,**kwargs):
        """
        Ranges for parameters:
            l_value:        any value desired for strength of penalty.
            non_zero_tol:   desired maximum non-zero weight percentage, for example .45
            zero_tol:       desired threshold under which to consider values 0, should be smaller than 1e-5
        """
        self.l_value = l_value
        self.apply = False
        self.non_zero_tol = non_zero_tol
        self.zero_tol = zero_tol
        self.verbose = verbose
    def __call__(self,output,side=[]):
        ratio_zero = (abs(output)<self.zero_tol).sum()/output.shape[0]
        penalty = (1-ratio_zero)
        if self.verbose >0:
            return max(0,penalty-self.non_zero_tol)*self.l_value,penalty
        return max(0,penalty-self.non_zero_tol)*self.l_value
    def oos(self,output,side=[]):
        return output
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["min_val", "l_value"]}
        pars["name"] = "weights_non_zero"
        return pars
    
class OBJ_P_mc_max(OBJ_P):
    def __init__(self,mc_factor=4,l_value=.1):
        self.mc_factor = mc_factor
        self.apply = True
        self.l_value = l_value
    def __call__(self,output,side):
        side = side/sum(side)*self.mc_factor*sum(abs(output))
        values = (abs(output)-side)
        values = values[values>0]
        return values.sum()*self.l_value
    def oos(self,output,side=[]):
        
        output = output/abs(output).sum() # standardized weights
        side = side/sum(side)*self.mc_factor # maximum share of weights
        factor = output/abs(output) #sign of weights
        series = abs(output)-side # weights - max_weights
        series[series<0] = 0
        while sum(series)>0:
            output[series>0] = 0
            replace = side+0
            replace[series==0] = 0
            output= output/abs(output).sum()*(1-replace.sum())
            replace*=factor
            output += replace
            series = abs(output)-side # weights - max_weights
            series[series<0] = 0
        return output
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["mc_factor", "l_value"]}
        pars["name"] = "mc_max"
        return pars
    
##################################################
################### Objectives ###################
##################################################

class Objective():
    def __init__(): pass
    def __call__(): pass
    def int_result(): pass
    def result(): pass
    def reset(): pass
    def clear(): pass
    def test_result(): pass
    @classmethod
    def from_dict(cls,pars):
        if pars["objective"] == "SDF":
            # pars["loss_fn"] = Loss_SDF
            return SDF_objective.from_dict(pars)
        elif pars["objective"] == "RP":
            # pars["loss_fn"] = Loss_ENET
            return RP_objective.from_dict(pars)
    
class RP_objective(Objective):
    def __init__(self,
                 loss_fn = Loss_ENET,
                 loss_fn_par = {"base_penalty":Loss_MSE_weighted(),
                                "alpha_p":0.00001,
                                "lambda_p":0.0000001,
                                "volatility" : False},
                 quantiles = 10,
                 target_is_return = True,
                 **kwargs
                 ):
        self.clear()
        self.quantiles = quantiles
        self.target_is_return = target_is_return
        self.loss_fn = loss_fn(**loss_fn_par)
        self.reset()
        self.thread = 0
    
    def reset(self):
        self.epochs_losses = {}; self.epochs_scores = {};
        self.int_result_timer = timeit(4)
        self.epoch_timer = timeit(4)
        self.new = True
        self.sub_loss, self.sub_len = 0, 0
        self.sub_rss, self.sub_tss = 0, 0
        self.sub_penalty = 0; self.loss = 0;
        
    def __call__(self, pred, y, side, model, optimizer=None,Stream = None):
        # import pdb; pdb.set_trace()
        # this is to slow
        # compute local loss
        if self.new:
            self.epoch_timer.reset()
            self.int_result_timer.reset()
            self.new = False
        # timer = timeit(10)
        try:
            with to.cuda.stream(Stream):
                self.loss, self.penalty = self.loss_fn(pred, y,side,model)
            pred = pred.detach()
        except:
            print("pred shape:",pred.shape,
                  "\ny shape:",y.shape,
                  "\npred type",type(y))
            raise
        # timer("loss_fn")
        # stopper = np.isnan(loss.item())
        self.sub_rss += ((pred-y)**2).sum()
        self.sub_tss += ((y)**2).sum()
        # import pdb; pdb.set_trace()
        self.sub_len += pred.shape[0]
        
        self.loss += self.penalty
        if optimizer is not None:
            with to.cuda.stream(Stream):
                # optimizer.zero_grad() # shifted from model class
                self.loss.backward()
                optimizer.step()
        self.loss       = self.loss.cpu().detach().numpy()
        self.penalty    = self.penalty.cpu().detach().numpy()
        self.loss -= self.penalty
        
        self.sub_loss += self.loss*pred.shape[0]
        self.sub_penalty += self.penalty*pred.shape[0]
        # timer("losses")
        # Backpropagation
        
        # timer("backpropagation")
    
    def int_result(self,batch,n_batches,print_option=True):
        # import pdb; pdb.set_trace()
        self.rss += self.sub_rss
        self.tss += self.sub_tss
        self.train_loss += self.sub_loss
        self.len+=self.sub_len
        self.train_penalty+=self.sub_penalty
        if print_option:
            determination = 1-self.sub_rss/self.sub_tss
            loss = (self.sub_loss)/self.sub_len
            penalty = self.sub_penalty/self.sub_len
        self.sub_loss, self.sub_len = 0, 0
        self.sub_rss, self.sub_tss = 0, 0
        self.sub_penalty = 0
        int_time = self.int_result_timer()
        if print_option:
            if loss == to.inf: import pdb; pdb.set_trace()
            with print_lock:
                print(f"th: {self.thread:2d} || ",end = "")
                print(f"loss : {loss: 15.13f} || penalty: {penalty: 12.8f} || R2: {determination: 12.8f} || ",end = "")
                print(f"subtime: {int_time:10.6f} || [{batch:>5d}/{n_batches:>5d}]")
    
    def result(self,msg):
        self.int_result(0,0,False)
        # if msg == "val":import pdb; pdb.set_trace()
        
        determination = 1-self.rss/self.tss
        loss = self.train_loss/self.len
        penalty = self.train_penalty/self.len
        epoch_time = self.epoch_timer()
        if msg in self.epochs_losses.keys():
            self.epochs_losses[msg].append(loss+penalty)
            self.epochs_scores[msg].append(determination)
        else:
            self.epochs_losses[msg] = [loss+penalty]
            self.epochs_scores[msg] = [determination]
        self.rss = 0; self.tss = 0; self.train_loss = 0; 
        self.len=0; self.train_penalty = 0;
        with print_lock:
            print(f"th: {self.thread:2d} || ",end = "")
            print(f"{msg:5s}: loss: {loss: 9.7f} || penalty: {penalty: 12.8f} || R2: {determination: 12.8f} || ",end = "")
            print(f"epoch_time: {epoch_time: 10.6f}")
        return loss, penalty
    
    def test_result(self, test, side = [], 
                    date_entity="day", 
                    target_periods=1, 
                    return_mode="log",
                    limit = None,
                    **kwargs):
        # import pdb; pdb.set_trace()
        if side == []:
            side=None
        elif type(side) == list:
            side = side[0]
        self.epochs_losses["test"] = sum((test["pred"]-test["test"])**2)/test["pred"].shape[0]
        quantiles = self.quantiles
        R2_mod = 1-sum((test["pred"]-test["test"])**2)/sum(test["test"]**2)
        R2 = 1-sum((test["pred"]-test["test"])**2)/sum((test["test"]-test["test"].mean())**2)
        self.epochs_scores["test"] = R2+0
        print_hint2("Prediction Results for Test:",1,1,"—",width_field=75)
        print(f"\tR2_oos           : {float(R2): 2.8f}, "+\
              f"\n\tR2_oos (modified): {float(R2_mod): 2.8f}, "+\
              f"\n\tloss             : {self.epochs_losses['test']: 2.10f}")
        test_predictive={"R2":R2,"R2_mod":R2_mod}
        MCA = QAA(test[["test","pred"]], date_entity = date_entity, quan=quantiles)
        print("QAA for iteration = ",MCA)
        test_predictive["QAA"]=MCA
        
        columns_pf = ["test"]
        if side is not None:
            columns_pf.append(side)
        
        test_portfolio = pd.DataFrame()
        if self.target_is_return:
            # end of prediction performance, change of quantiles by target periods holding
            if return_mode == "log":
                test["test"] = np.exp(test["test"])-1
            test[["test","pred"]]/=target_periods
            test["test"] = test["test"].apply(lambda x: (x+1)**(1/target_periods))-1
            
            # if limit != None: # limit for number of stocks in portfolio
            #     upper_limit = frame.nlargest(int(limit/2),"pred")["pred"].values[int(limit/2)-1]
            #     lower_limit = frame.nsmallest(int(limit/2),"pred")["pred"].values[int(limit/2)-1]
            #     frame.loc[(frame["pred"]<upper_limit) & (frame["pred"]>lower_limit)] = 0
            quantiles*=target_periods
            deciles = pd.DataFrame(columns=["test"])
            for quantile in range(1,quantiles):
                test = test.join(test["pred"].groupby(date_entity).quantile(q=(1/quantiles)*quantile).\
                                 to_frame(name="dec"+str(quantile)),on=date_entity,how="inner")
                if quantile==1:
                    continue
                # if return_mode == "simple":
                deciles.loc["dec"+str(quantile),"test"] =\
                    (1+test[(test["pred"]>test["dec"+str(quantile-1)])&\
                          (test["pred"]<=test["dec"+str(quantile)])]["test"]).\
                        groupby(date_entity).mean().prod()
                            
                # elif return_mode == "log":
                #     deciles.loc["dec"+str(quantile),"test"] =\
                #        np.exp(test[(test["pred"]>test["dec"+str(quantile-1)])&\
                #               (test["pred"]<=test["dec"+str(quantile)])]["test"]).\
                #             groupby(date_entity).mean().prod()
                if quantile == quantiles-1:
                    name_2max = "dec"+str(quantile)
                    name_max = "dec"+str(quantile+1)
            # if return_mode == "simple":
            deciles.loc["dec1"] = (1+test[test["pred"]<=test["dec1"]]["test"]).\
                groupby(date_entity).mean().prod()
            deciles.loc[name_max] = (1+test[test["pred"]>test[name_2max]]["test"]).\
                groupby(date_entity).mean().prod()
            # elif return_mode == "log":
            #     deciles.loc["dec1"] = np.exp(test[test["pred"]<=test["dec1"]]["test"]).\
            #         groupby(date_entity).mean().prod()
            #     deciles.loc[name_max] = np.exp(test[test["pred"]>test[name_2max]]["test"]).\
            #         groupby(date_entity).mean().prod()
            # deciles.sort_values(str(col),inplace=True,ascending=False)
            low_port = test[test["pred"]<=test["dec1"]]["test"]
            high_port = test[test["pred"]>test[name_2max]]["test"]
            # if return_mode == "simple":
            high        = (high_port).groupby(date_entity).mean()
            low         = (low_port).groupby(date_entity).mean()
            std         = test["test"].groupby(date_entity).std()
            
            avg         = (test["test"]).groupby(date_entity).mean()
            
            HML_std     = pd.concat(
                [(low_port-low_port.groupby(date_entity).mean()),
                (high_port-high_port.groupby(date_entity).mean())]
                ).groupby(date_entity).std()
            high_std    = low_port.groupby(date_entity).std()
            low_std     = high_port.groupby(date_entity).std()
            # elif return_mode == "log":
            #     high_port   = np.exp(high_port)
            #     low_port    = np.exp(low_port)
            #     high        = high_port.groupby(date_entity).mean()
            #     avg         = np.exp(test["test"]).groupby(date_entity).mean()
            #     low         = low_port.groupby(date_entity).mean()
                
            #     std         = np.exp(test["test"]).groupby(date_entity).std()
            #     HML_std     = pd.concat(
            #         [(low_port-low_port.groupby("month").mean()),
            #         (high_port-high_port.groupby("month").mean())]
            #         ).groupby(date_entity).std()
            #     high_std    = (high_port-high_port.groupby("month").mean()).groupby(date_entity).std()
            #     low_std     = (low_port-low_port.groupby("month").mean()).groupby(date_entity).std()
            del high_port, low_port
            
            # portfolio by date entity
            test_portfolio = pd.DataFrame(
                {"sample_avg":avg,
                 "HML":high-low,
                 "high":high,
                 "low":low,
                 "sample_std":std,
                 "HML_std": HML_std,
                 "high_std":high_std,
                 "low_std":low_std,})
            test_portfolio.loc["aggregate",:] = [
                (avg+1).prod()-1,
                (high-low+1).prod()-1,
                (high+1).prod()-1,
                (low+1).prod()-1,
                avg.std(),
                (high-low).std(),
                high.std(),
                low.std()
                ]
            print_hint2("Portfolio Results for Test:",1,1,"—",width_field=75)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.width', 125):
                print(test_portfolio.round(4).loc["aggregate",:].to_frame().T)
        
        pred_stats = pd.DataFrame({
            "avg_pred":test["pred"].groupby(date_entity).mean(),
            "l_ratio":test.loc[test["pred"]>0,"pred"].groupby(date_entity).count()/\
                (test["pred"].groupby(date_entity).count()).replace(np.nan,0),
            "s_ratio":test.loc[test["pred"]<0,"pred"].groupby(date_entity).count()/\
                (test["pred"].groupby(date_entity).count()).replace(np.nan,0),
            "std_pred":test["pred"].groupby(date_entity).std(),
            "min_pred":test["pred"].groupby(date_entity).min(),
            "max_pred":test["pred"].groupby(date_entity).max(),
            "skewness":test["pred"].groupby(date_entity).skew(),
            "kurtorsis":test["pred"].groupby(date_entity).apply(pd.Series.kurt),
            })
        pred_stats.loc["aggregate",:] = [
            test["pred"].mean(),
            test.loc[test["pred"]>0,"pred"].count()/\
                (test["pred"].count()),
            test.loc[test["pred"]<0,"pred"].count()/\
                (test["pred"].count()),
            test["pred"].std(),
            test["pred"].min(),
            test["pred"].max(),
            test["pred"].skew(),
            test["pred"].kurtosis()
            ]
        columns_pf.append("pred")
        return test_predictive, test_portfolio, test[columns_pf], pred_stats
    
    def clear(self):
        self.len,self.sub_len = 0,0
        self.train_loss = 0
        self.train_penalty = 0
        self.sub_rss, self.sub_tss = 0, 0
        self.rss, self.tss = 0, 0
        self.sub_loss = 0
        self.sub_penalty = 0
    
    def serialise(self):
        pars = {"objective":"RP"}
        if type(self.loss_fn) is Loss_ENET:
            pars["loss_fn"] = "Loss_ENET"
            pars["loss_fn_par"] = {
                "base_penalty": "MSE_Loss",
                "alpha_p":self.loss_fn.alpha_p,
                "lambda_p":self.loss_fn.lambda_p}
            pars["quantiles"] = self.quantiles
        return pars
    
    @classmethod
    def deserialise(cls,pars):
        if pars["loss_fn"] == "Loss_ENET" or pars["loss_fn"] == "Loss_elastic_net":
            pars["loss_fn"] = Loss_ENET
            pars["loss_fn_par"]["base_penalty"] = Loss_MSE_weighted()
        return pars
    
    @classmethod
    def from_dict(cls,pars):
        pars = cls.deserialise(pars)
        return cls(**pars)
    
    
class SDF_objective(Objective):
    def __init__(self,
                 loss_fn = Loss_SDF,
                 loss_fn_par = {"normalise":"1",
                                "return_mode":"log"},
                 quantiles = 10,
                 **kwargs):
        self.clear()
        self.quantiles = quantiles
        self.loss_fn = loss_fn(**loss_fn_par)
        self.reset()
        self.thread = 0
        
    def reset(self):
        self.epochs_losses = {}; self.epochs_scores = {};
        self.int_result_timer = timeit(4)
        self.epoch_timer = timeit(4)
        self.new = True
        self.sub_score, self.sub_len = 0, 0
        self.sub_penalty = 0
        
    def __call__(self,pred,y,side,model,optimizer,Stream=None):
        # import pdb; pdb.set_trace()
        if self.new:
            self.epoch_timer.reset()
            self.int_result_timer.reset()
            self.new = False
        # timer = timeit(10)
        try:
            with to.cuda.stream(Stream):
                self.loss, self.penalty = self.loss_fn(pred, y,side)
                # print("loss_fn:         ",timer("loss_fn"))
                pred = pred.detach()
        except:
            print("pred shape:",pred.shape,
                  "\ny shape:",y.shape,
                  "\npred type",type(y))
            raise
        # stopper = np.isnan(loss.item())
        self.loss += self.penalty
        if optimizer is not None:
            with to.cuda.stream(Stream):
                # optimizer.zero_grad() # shifted from model class
                self.loss.backward()
                # print("loss backward:   ",timer("loss_backward"))
                optimizer.step()
        self.loss -= self.penalty
        self.loss       = self.loss.cpu().detach().numpy()
        self.penalty    = self.penalty
        self.sub_score   += self.loss*pred.shape[0]
        self.sub_len     += pred.shape[0]
        self.sub_penalty += self.penalty*pred.shape[0]
        # print("losses:          ",timer("losses"))
        # Backpropagation
        
    def int_result(self,batch,n_batches,print_option=True):
        self.train_loss += self.sub_score
        self.train_penalty += self.sub_penalty
        self.len+=self.sub_len
        if print_option:
            SDF = self.sub_score/self.sub_len
            penalty = self.sub_penalty/self.sub_len
        self.sub_score, self.sub_len = 0, 0
        self.sub_penalty = 0
        int_time = self.int_result_timer()
        if print_option:
            with print_lock:
                print(f"th: {self.thread:2d} || ",end = "")
                print(f"SDF  : {SDF:1.7f} || penalty: {penalty:4.10f} ||",end = "")
                print(f"int_time: {int_time:3.9f} || [{batch:>2d}/{n_batches:>2d}]")
                
    def result(self,msg):
        self.int_result(0,0,False)
        
        SDF = self.train_loss/self.len
        penalty = self.train_penalty/self.len
        if msg in self.epochs_losses.keys():
            self.epochs_losses[msg].append(SDF+penalty)
            self.epochs_scores[msg].append(SDF+penalty)
        else:
            self.epochs_losses[msg] = [SDF+penalty]
            self.epochs_scores[msg] = [SDF+penalty]
        self.train_loss = 0; self.train_penalty = 0; self.len = 0;
        epoch_time = self.epoch_timer()
        with print_lock:
            print(f"th: {self.thread:2d} || ",end = "")
            print(f"{msg:5s}: SDF: {SDF:1.7f} || penalty: {penalty:4.10f} ||",end = "")
            print(f"epoch_time: {epoch_time:3.9f}")
        return SDF, penalty
    
    def test_result(self, test, side=None, 
                    date_entity="day", 
                    target_periods=1, 
                    return_mode="log",
                    limit = None # maximum number of stocks in portfolio
                    ):
        # import pdb; pdb.set_trace()
        if side == []:
            side=None
        elif type(side) == list:
            side = side[0]
        quantiles = self.quantiles
        # return_mode = self.loss_fn.return_mode    
        
        print_hint2("Prediction Results for Test:",1,1,"—",width_field=75)
        MCA = QAA(test[["test","pred"]], date_entity = date_entity, quan=quantiles)
        print(f"QAA for iteration = {MCA}",)
        test_predictive = {"QAA":MCA}
        
        
        # end of prediction performance, change of quantiles by target periods holding
        if return_mode == "log":
            test[["test"]]/=target_periods
        else:
            test["test"] = test["test"].apply(lambda x: (x+1)**(1/target_periods))-1
            # test["pred"] = test["pred"].apply(lambda x: (x+1)**(1/target_periods))-1
            
        SDF_w_por = []
        new_test = []
        for date in test.index.get_level_values(date_entity).unique():
            # ensure settings for sum of weights
            frame = test[test.index.get_level_values(date_entity) == date]
            if limit != None: # limit for number of stocks in portfolio
                upper_limit = frame.nlargest(int(limit/2),"pred")["pred"].values[int(limit/2)-1]
                lower_limit = frame.nsmallest(int(limit/2),"pred")["pred"].values[int(limit/2)-1]
                frame.loc[(frame["pred"]<upper_limit) & (frame["pred"]>lower_limit)] = 0
            frame["pred"] = self.loss_fn.norm_pred(
                to.tensor(frame["pred"].values)) 
            output = to.tensor(frame["pred"].values)
            target = to.tensor(frame["test"].values)
            side_values = []
            if side is not None:
                side_values = to.tensor(frame[side].values)
            SDF_w_por.append(1-self.loss_fn(output,target,side_values,apply = True,batch_size=1)\
                             [0].cpu().detach().numpy())
            new_test.append(frame)
        test = pd.concat(new_test)
        del new_test
        self.epochs_losses["test"] = 1-np.mean(SDF_w_por)
        self.epochs_scores["test"] = 1-np.mean(SDF_w_por)
        
        print(f"\tSDF          : {float(self.epochs_scores['test']): 2.8f}")
        
        if return_mode == "log":
            test["test"] = np.exp(test["test"])-1
        
        quantiles*=target_periods
        deciles = pd.DataFrame(columns=["test"])
        columns_pf = ["test"]
        if side is not None:
            columns_pf.append(side)
        for quantile in range(1,quantiles):
            # quantile = 2
            ### creating quantile border for the respective quantile
            test = test.join(test["pred"].groupby(date_entity).quantile(q=(1/quantiles)*quantile).\
                             to_frame(name="dec"+str(quantile)),on=date_entity,how="inner")
            if quantile==1:
                continue
            # portfolio return for respective decile portfolio, depending on return_mode:
            decile_returns = test[(test["pred"]>test["dec"+str(quantile-1)])&
                  (test["pred"]<=test["dec"+str(quantile)])][columns_pf]
            # if return_mode == "log":
            #     decile_returns["test"] = np.exp(decile_returns["test"])-1
            if side is None:
                ## without weighting series: simple average of decile returns
                deciles.loc["dec"+str(quantile),"test"] =\
                    (1+decile_returns["test"]).\
                    groupby(date_entity).mean().prod()
            else:
                ## with weighting series: weighted average of decile returns:
                deciles.loc["dec"+str(quantile),"test"] =\
                    ((decile_returns["test"]*decile_returns[side]).\
                        groupby(date_entity).sum()/\
                            decile_returns[side].groupby(date_entity).sum()+1).prod()
            if quantile == quantiles-1:
                name_2max = "dec"+str(quantile)
                name_max = "dec"+str(quantile+1)
        ## low_portfolio
        decile_returnsL = test[test["pred"]<=test["dec1"]][columns_pf]
        # if return_mode == "log":
        #     decile_returnsL["test"] = np.exp(decile_returnsL["test"])-1
        if side is None:
            ## without weighting series: simple average of decile returns
            low_port = (decile_returnsL["test"]).\
                groupby(date_entity).mean()+1
            deciles.loc["dec1","test"] = low_port.prod()
        else:
            ## with weighting series: weighted average of decile returns:
            low_port = (decile_returnsL["test"]*decile_returnsL[side]).\
                groupby(date_entity).sum()/\
                    decile_returnsL[side].groupby(date_entity).sum()+1
            deciles.loc["dec1","test"] =low_port.prod()
         
        ## high portfolio
        decile_returnsH = test[test["pred"]>=test[name_2max]][columns_pf]
        # if return_mode == "log":
        #     decile_returnsH["test"] = np.exp(decile_returnsH["test"])-1
        if side is None:
            ## without weighting series: simple average of decile returns
            high_port = decile_returnsH["test"].\
                groupby(date_entity).mean()+1
            deciles.loc[name_max,"test"] =high_port.prod()
        else:
            ## with weighting series: weighted average of decile returns:
            high_port = (decile_returnsH["test"]*decile_returnsH[side]).\
                groupby(date_entity).sum()/\
                    decile_returnsH[side].groupby(date_entity).sum()+1
            deciles.loc[name_max,"test"] =high_port.prod()
        
        # market comparison
        returns_total = copy.deepcopy(test[columns_pf])
        # if return_mode == "log":
        #     returns_total["test"] = np.exp(returns_total["test"])-1
        if side is None:
            avg_port = returns_total["test"].groupby(date_entity).mean()+1
        else:
            avg_port = (returns_total["test"]*returns_total[side]).\
                groupby(date_entity).sum()/returns_total[side].\
                    groupby(date_entity).sum()+1
        del returns_total
        
        
        # deciles.sort_values(str(col),inplace=True,ascending=False)
        # low_port = test[test["pred"]<=test["dec1"]]["test"]
        # high_port = test[test["pred"]>test[name_2max]]["test"]
        # avg         = (avg_port+1).groupby(date_entity).mean()
        # low         = (low_port+1).groupby(date_entity).mean()
        
        std         = test["test"].groupby(date_entity).std()
        HML_std     = pd.concat(
            [(low_port-low_port.groupby(date_entity).mean()),
             (high_port-high_port.groupby(date_entity).mean())]
            ).groupby(date_entity).std()
        high_std    = decile_returnsH["test"].groupby(date_entity).std()
        low_std     = decile_returnsL["test"].groupby(date_entity).std()
        # del high_port, low_port
        
        # portfolio by date entity
        test_portfolio = pd.DataFrame(
            {"sample_avg":avg_port,
             "SDF": SDF_w_por,
             "HML":high_port-low_port,
             "high":high_port,
             "low":low_port,
             "sample_std":std,
             "HML_std": HML_std,
             "high_std":high_std,
             "low_std":low_std,})
        test_portfolio.loc["aggregate",:] = [
            avg_port.prod()-1,
            np.prod(np.array(SDF_w_por)+1)-1,
            (high_port-low_port+1).prod()-1,
            high_port.prod()-1,
            low_port.prod()-1,
            avg_port.std(),
            (high_port-low_port).std(),
            high_port.std(),
            low_port.std()
            ]
        print_hint2("Portfolio Results for Test:",1,1,"—",width_field=75)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.width', 125):
            print(test_portfolio.round(4).loc["aggregate",:].to_frame().T)
        # import pdb; pdb.set_trace()
        pred_stats = pd.DataFrame({
            "sum_weights":test["pred"].apply(abs).groupby(date_entity).sum(),
            "sum_neg_weights":test.loc[test["pred"]<0,"pred"].groupby(date_entity).sum(),
            "sum_pos_weights":test.loc[test["pred"]>0,"pred"].groupby(date_entity).sum(),
            "l_ratio":test.loc[test["pred"]>0,"pred"].groupby(date_entity).count()/\
                (test["pred"].groupby(date_entity).count()).replace(np.nan,0),
            "s_ratio":test.loc[test["pred"]<0,"pred"].groupby(date_entity).count()/\
                (test["pred"].groupby(date_entity).count()).replace(np.nan,0),
            "std_weights":test["pred"].groupby(date_entity).std(),
            "min_weights":test["pred"].groupby(date_entity).min(),
            "max_weights":test["pred"].groupby(date_entity).max(),
            "skewness":test["pred"].groupby(date_entity).skew(),
            "kurtorsis":test["pred"].groupby(date_entity).apply(pd.Series.kurt),
            })
        
        pred_stats.loc["aggregate",:] = [
            test["pred"].abs().sum()/len(high_port),
            test.loc[test["pred"]<0,"pred"].sum(),
            test.loc[test["pred"]>0,"pred"].sum(),
            test.loc[test["pred"]>0,"pred"].count()/\
                (test["pred"].count()),
            test.loc[test["pred"]<0,"pred"].count()/\
                (test["pred"].count()),
            test["pred"].std(),
            test["pred"].min(),
            test["pred"].max(),
            test["pred"].skew(),
            test["pred"].kurtosis()
            ]
        columns_pf.append("pred")
        return test_predictive, test_portfolio, test[columns_pf], pred_stats
    
    def clear(self):
        self.len,self.sub_len   = 0,0
        self.train_loss         = 0
        self.train_penalty      = 0
        self.sub_score          = 0
        self.sub_penalty        = 0
        
    def serialise(self):
        # import pdb; pdb.set_trace()
        pars = {"objective":"SDF"}
        if type(self.loss_fn) is Loss_SDF:
            pars["loss_fn"] = "Loss_SDF"
            pars["loss_fn_par"] = {attr:getattr(self.loss_fn,attr) for attr in vars(self.loss_fn) if attr in\
                    ["return_mode","penalties","mode","lambda_s","normalise"]}
            for penalty_i in range(len(pars["loss_fn_par"]["penalties"])):
                pars["loss_fn_par"]["penalties"][penalty_i] = \
                    pars["loss_fn_par"]["penalties"][penalty_i].serialise()
            pars["quantiles"] = self.quantiles
        return pars
    
    @classmethod
    def deserialise(cls,pars):
        for penalty_n in range(len(pars["loss_fn_par"]["penalties"])):
            pars["loss_fn_par"]["penalties"][penalty_n] = \
                OBJ_P.from_dict(pars["loss_fn_par"]["penalties"][penalty_n])
        if pars["loss_fn"] == "Loss_SDF":
            pars["loss_fn"] = Loss_SDF
        return pars
    
    @classmethod
    def from_dict(cls,pars):
        pars = cls.deserialise(pars)
        return cls(**pars)

################################################################################
################################################################################
##########################  Scheduling and Monitoring  #########################
################################################################################
################################################################################

def np_rolling_mean(array1d,window_length):
    array1d = pd.DataFrame(array1d,columns = ["value"])
    array1d["mean"] = array1d["value"].rolling(window_length,1).mean()
    means = array1d["mean"].to_numpy()
    return means


class Scheduler_Monitor():
    def __init__(self,n_epochs:float=.5,
                 decline:str="linear",
                 min_delta:float=0.0001,
                 max_declines = 5,
                 n_epochs_no_change = 2.5,
                 skip_batches:int = 0,
                 min_delta_relative=False,
                 epoch_schedule = True, ## wether to decline on validation data
                 **kwargs):
        '''
        n_epochs : float
            fraction of epochs after which a learning rate decay happens.
        decline : str, ONE OF
            'linear': 
                performs scheduler as is
            'adaptive': 
                performs scheduler if score has not reduced for n_epochs*batch_per_epoch 
                number of batches
        min_delta : float
            minimum change of score to trigger change in learning rate
        '''
        self.skip_batches = skip_batches
        self.n_epochs = n_epochs
        self.decline = decline
        self.epoch_schedule = epoch_schedule
        self.min_delta_relative = min_delta_relative
        self.min_delta = min_delta
        self.max_declines = max_declines
        self.reset()
        self.n_epochs_no_change = n_epochs_no_change
        for key, value in kwargs.items():
            setattr(self,key,value)
            
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.declines = 0
        self.batch_min = 0
        self.learning_rates = []
        self.val_last_loss = np.nan
        self.iter_no_change = 0
        self.old_mean = np.inf
        
    def set_pars(self,n_batches,shuffle=False):
        self.n_batches = float(n_batches)
        self.change_epochs = int(self.n_batches*self.n_epochs)
        if not self.epoch_schedule:
            self.n_batches_no_change = int(self.n_epochs_no_change*n_batches)
        else:
            self.n_batches_no_change = self.n_epochs_no_change+0
        
        if shuffle:
            self.old_mean = np.inf
            # self.iter_no_change = 0
        
    def train(self,scheduler,loss):
        self.train_losses.append(float(loss))
        self.val_losses.append(self.val_last_loss)
        self.learning_rates.append(scheduler.get_last_lr()[0])
        if not self.epoch_schedule:
            self.evaluate(self.train_losses,scheduler)
                    
    def val(self,scheduler,loss):
        self.val_losses[-1] = float(loss)+0
        self.val_last_loss = float(loss)+0
        if self.epoch_schedule:
            return self.evaluate(self.val_losses,scheduler)
            
    def evaluate(self,losses,scheduler):
        if self.decline == "linear":
            self.iter_no_change +=1
            # print("val triggered:",self.iter_no_change)
            if self.iter_no_change>=self.n_batches_no_change and \
                self.declines < self.max_declines:
                print(f"\nScheduled learning rate change after {self.n_batches_no_change} epochs",
                      "\nbatch (epoch) = {:7d} ({:7.3f})".format(
                          len(self.train_losses),
                          len(self.train_losses)/self.n_batches),
                      "\nbatch_min (epoch) = {:7d} ({:7.3f})".format(
                          self.batch_min,
                          self.batch_min/self.n_batches),"\n")
                scheduler.step()
                self.iter_no_change = 0
                self.declines+=1
                return True
        elif self.decline == "adaptive":
            if len(losses)<self.change_epochs+self.skip_batches:
                pass
            elif len(losses)==self.change_epochs+self.skip_batches:
                ## initiate old_mean
                self.old_mean = np.mean(losses[-self.change_epochs:])#min(,max(self.skip_batches,.5))
            else:
                new_mean = np.mean(losses[-self.change_epochs:])
                if not self.min_delta_relative:
                    min_delta_new_min =new_mean+self.min_delta 
                else:
                    min_delta_new_min =new_mean*(1+self.min_delta)
                if self.old_mean>min_delta_new_min:
                    self.iter_no_change = 0
                    self.batch_min = len(self.train_losses)+0
                    self.old_mean = new_mean+0
                else:
                    self.iter_no_change += 1
                    if self.iter_no_change>=self.n_batches_no_change and\
                        self.declines<self.max_declines:
                        self.declines+=1
                        print("\nChange in learning rate:\nold_mean={:9.7f}".format(self.old_mean),
                              "  new_mean={:9.7f}".format(min_delta_new_min),
                              "  batch (epoch) = {:7d} ({:7.3f})".format(
                                  len(self.train_losses),
                                  len(self.train_losses)/self.n_batches),
                              "  batch_min (epoch) = {:7d} ({:7.3f})".format(
                                  self.batch_min,
                                  self.batch_min/self.n_batches),"\n")
                        scheduler.step()
                        self.iter_no_change=0
                        return True
        return False
                
    def plot(self,plot_batches = True,cap=None):
        n_batches = self.n_batches
        losses = pd.DataFrame()
        losses["train_raw"] = np.array(self.train_losses[self.skip_batches:])
        if cap is not None:
            losses["train_raw"] = losses["train_raw"].clip(upper=cap)
        losses["train_smoothed"] = losses["train_raw"].rolling(self.change_epochs,1).mean()
        losses["val"] = self.val_losses[self.skip_batches:]
        losses["lr"] = self.learning_rates[self.skip_batches:]
        losses = losses.apply(np.log)
        
        fig, ax = plt.subplots(figsize = (8,4.5),dpi=200)
        fig.patch.set_facecolor("white")
        ax.patch.set_facecolor("white")
        fig.suptitle('Model losses', fontsize=14, fontweight='bold',color="Black")
        ax2 = ax.twinx()
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        
        if plot_batches:
            losses["train_raw"].plot(ax=ax, style = "-", rot = 30,
                                     label="batch losses", linewidth = .5, zorder=2)
        losses["train_smoothed"].plot(ax=ax, style='-',rot = 30,label="losses rolling mean",linewidth = 2,
                                      zorder=2)
        losses["val"].plot(ax=ax, style='-',rot = 30,label="losses validation",linewidth = 2,
                           color = "purple", zorder=2)
        losses["lr"].plot(ax=ax2, style = "-", rot = 30, label="learning rate",
                          linewidth = 2, color = "red", zorder=2)
        ### some kind of index calculations with respect to the epoch
        
        # ax.legend(loc="best",bbox_to_anchor=(-0.075,1.1,0.15,0.1))
        # # ax.imshow(phione_neg)
        # ax2.legend(loc="best",bbox_to_anchor=(0.925,1.1,0.15,0.1))
        fig.legend(bbox_to_anchor=(0.755,.79,0.15,0.1))
        ax.tick_params(axis='x',colors="black")
        ax2.tick_params(axis='y',colors="red")
        ax.set_xlabel("Epoch", fontweight='bold',fontsize = 12.5,color="black")
        
        index_factor = (losses.shape[0]//n_batches-5)//20+1
        index_int = np.arange(-self.skip_batches,losses.shape[0],n_batches*index_factor)
        index_int[0] = 0
        index_str = np.arange(0,losses.shape[0]//n_batches+1,index_factor)
        ax.set_xticks(index_int)
        ax.set_xticklabels(index_str,rotation = 30)
        ax.set_ylabel(r"Log Losses", fontweight='bold',fontsize = 12.5,color="black")
        ax2.set_ylabel(r"Log Learning Rate", fontweight='bold',fontsize = 12.5,color="red")
        ax.grid(zorder=2)
    
    def serialise(self):
        pars = {attr:getattr(self,attr) for attr in vars(self) if attr in\
                ["n_epochs","decline","min_delta","max_declines","n_epochs_no_change",
                 "skip_batches","train_losses","val_losses","declines","batch_min",
                 "learning_rates","val_last_loss","iter_no_change","old_mean",
                 "n_batches","change_epochs","epoch_schedule","n_batches_no_change"]}
        return pars
    
    @classmethod
    def from_dict(cls,pars):
        return cls(**pars)
    
    def save(self,path):
        pars = self.serialise()
        with open(path,"w") as fp:
            json.dump(pars,fp,indent = 4)
            
    @classmethod
    def load(cls,path):
        with open(path,"rb") as fp:
            pars = json.load(fp)
        return cls(**pars)
    
################################################################################
################################################################################
############################  Model Container Class  ###########################
################################################################################
################################################################################

class model():
    def __init__(self,
                 model_type = CNN_v1, # model type
                 structure = CNN_v1_structure, # layers
                 learning_rate:float = 1e-3, # initial learning rate
                 objective = RP_objective(),
                 Scheduler_Monitor: Scheduler_Monitor=Scheduler_Monitor(.5,"linear",1e-4),
                 side = [],
                 scheduler = to.optim.lr_scheduler.StepLR,
                 scheduler_params = {"step_size":5,
                                     "gamma":.02,
                                     "last_epoch":- 1, 
                                     "verbose":False},
                 max_iter_no_change = 10,
                 betas = (.9,.999),
                 tolerance = 0.00001,
                 verbose = 1,
                 shuffle_epochs = 5,
                 device = None,
                 recover_weights_on_lr_decline = True,
                 path = "",
                 replace_path = False,
                 **kwargs
                  ):
        self.multifreq = False
        self.model_type = model_type
        self.structure = structure
        if model_type in [ANN_MultiFreq]:
            self.multifreq = True
        
        self.device = device
        self.path = path
        # self.thread = thread
        self.replace_path = replace_path
        self.recover_weights_on_lr_decline = recover_weights_on_lr_decline
        self.objective= objective
        self.Scheduler_Monitor = copy.deepcopy(Scheduler_Monitor)
        self.Stream = None# to.cuda.Stream()
        self.learning_rate = learning_rate
        self.betas = betas
        self.scheduler_function = scheduler
        self.scheduler_params=scheduler_params
        self.max_iter_no_change = max_iter_no_change
        self.tolerance = tolerance
        self.side = side
        self.vram_analysis = []
        self.shuffle_epochs = shuffle_epochs
        self.verbose = verbose
        self.reset()
        # rolling score over certain nr of batches or epochs:
        
        if path != "" and os.path.exists(path) and not replace_path:
            self.load(path)
            
    def reset(self):
        try:
            if self.device is None:
                global devices
                self.device = devices.pop()
                devices.append(self.device)
            self.model = self.model_type(self.structure,self.verbose).to(self.device)
        except TypeError:
            print("Class model: Structure is incompatible with model type or has\
                  \nincorrectly defined elements:")
            print(self.structure.structure)
            raise
        
        self.optimizer = to.optim.Adam(self.model.parameters(), lr=self.learning_rate+0,
                                       betas=self.betas)
        self.scheduler = self.scheduler_function(self.optimizer, 
                                   **self.scheduler_params)
        self.Scheduler_Monitor.reset()
        self.model_classification = self.model.model_classification
        
        self.best_score = np.inf
        self.best_epoch = 0
        self.best_score_no_change = np.inf
        self.epoch = 0
        self.timer = timeit(8)
        self.rolling_score = []
        self.test_results = {}
        self.test_data = pd.DataFrame()
        self.objective.reset()
        
    def save(self,path):
        to.save(self.model.state_dict(), f"{path}")
        
    def load(self,path):
        self.model.load_state_dict(to.load(f"{path}"))
        
    def get_model_type(self):
        # self.model_type = type(self.model)
        if to.distributed.is_initialized():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = toDDP(self.model, device_ids=[self.device],
                                output_device=self.device
                                )
            self.model_type = type(self.model.module)
            
    def train_validate(self,
                       data_set, #dataset or PhD4_dataset
                       n_epochs:int= 0,
                       to_epoch:int= 0,
                       fraction = 1,
                       n_parallel = 1,
                       thread = 0,
                       multip = False,
                       frequency = "m",
                       tolerance_relative = False):
        # import pdb; pdb.set_trace()
        data_set.model_log(thread)
        self.get_model_type()
        self.instrumented = False
        if self.model_type in [AE_1d]:
            if to.distributed.is_initialized():
                self.instrumented = self.model.module.instrumented
            else:
                self.instrumented = self.model.instrumented
        # else:
        # self.model = self.model.to(devices[thread%len(devices)])
        self.model.train() #set model to training mode for dropout layers.
        training_start_time = time.time()
        if self.verbose>0: self.timer("tvstart")
        if self.epoch == 0:
            self.Scheduler_Monitor.set_pars(
                data_set.props[frequency]["train"]["n_batches"])
        # print("to_epoch: {};\tepoch: {}".format(to_epoch,self.epoch))
        if to_epoch>self.epoch:
            print("Found max_epoch higher than model epoch:{me:d}>{e:d}".\
                  format(me=to_epoch,
                         e=self.epoch))
            n_epochs = to_epoch-self.epoch
        elif n_epochs==0:
            print("Found zero epochs.")
            return self.best_score
        if thread == 0:
            data_set.shuffle_iteration = self.epoch//self.shuffle_epochs
        tol = self.tolerance
        iter_since_change = 0
        best_weights = copy.deepcopy(self.model.state_dict())
        if self.verbose>0: self.timer("tvpre")
        for epoch in range(n_epochs):
            # data_set.reset_batch_count() # deactivated because if the fraction is not one, more parts of the data are used
            self.epoch += 1
            print("Epoch {:d}".format(self.epoch))
            if self.shuffle_epochs>0:
                if self.epoch%self.shuffle_epochs == 0:
                    shuffle_iteration = self.epoch//self.shuffle_epochs
                    if thread ==0:
                        while shuffle_iteration > data_set.model_get_status()+1:
                            time.sleep(1)
                        data_set.shuffle()
                        data_set.model_log_add(thread)
                        self.best_score = np.inf
                        self.best_score_no_change = np.inf
                        self.Scheduler_Monitor.set_pars(
                            data_set.get_props(frequency,"train")["n_batches"],
                            shuffle=True)
                        print_hint2("Shuffling data after "+str(self.epoch)+" epochs",2,2)
                        if self.verbose>0: self.timer("train_shuffle")
                    else: # thread != 0
                        while data_set.shuffle_iteration < shuffle_iteration:
                            time.sleep(1)
                        data_set.model_log_add(thread)
                        self.best_score = np.inf
                        self.best_score_no_change = np.inf
                        self.Scheduler_Monitor.set_pars(
                            data_set.get_props(frequency,"train")["n_batches"],
                            shuffle=True)
            self.model.train()
            if multip:
                tomp.spawn(self.train_loop,args=(data_set,fraction,n_parallel,thread,frequency),
                           nprocs = len(devices))
            else:
                self.train_loop(
                    data_set = data_set,
                    fraction = fraction, 
                    n_parallel = n_parallel,
                    thread = thread,
                    frequency = frequency)
            epoch_loss, epoch_penalty = self.val_loop(
                data_set = data_set,
                frequency = frequency)
            # print(to.cuda.memory_allocated())
            epoch_loss += epoch_penalty
            self.model.eval()
            
            # import pdb; pdb.set_trace()
            lr_decline = self.Scheduler_Monitor.val(
                self.scheduler, epoch_loss+0)
            
            if lr_decline and self.recover_weights_on_lr_decline:
                print("Recovered best weights upon lr decline!")
                self.model.load_state_dict(best_weights)
            if self.verbose>0: self.timer("sched_val")
            iter_since_change += 1
            ## calculate benchmark score
            tolerance_score = self.best_score_no_change-tol
            if tolerance_relative:
                tolerance_score = self.best_score_no_change*(1-tol)
                
            if epoch_loss < tolerance_score or \
                (epoch == self.max_iter_no_change/2 and max(n_epochs,to_epoch)>self.max_iter_no_change*1.5):
                self.best_score_no_change   = copy.deepcopy(epoch_loss)
                self.best_score             = copy.deepcopy(epoch_loss)
                best_epoch_no_change        = self.epoch+0
                self.best_epoch             = self.epoch+0
                del best_weights; to.cuda.empty_cache();
                best_weights                = copy.deepcopy(self.model.state_dict())
                iter_since_change           = 0
            elif epoch_loss < self.best_score:
                self.best_score             = copy.deepcopy(epoch_loss)
                self.best_epoch             = self.epoch+0
                del best_weights; to.cuda.empty_cache();
                best_weights                = copy.deepcopy(self.model.state_dict())
            if iter_since_change == self.max_iter_no_change:
                print(("\nNo tolerance change for {minc:4d} epochs at loss of {bc:12.8f} "+\
                      "in epoch {epoch:4d}.").\
                      format(minc = self.max_iter_no_change,
                              bc = self.best_score_no_change,
                              epoch = best_epoch_no_change))
                if self.verbose>0: self.timer("epoch_end")
                break
            if self.verbose>0: self.timer("epoch_end")
        
        print("\nBest score in epoch {epoch:4d} at {bsc:12.8f}.\n".format(
            epoch = self.best_epoch, bsc = self.best_score))
        self.model.load_state_dict(best_weights)
        if self.verbose >0: print("Process {} completed loading state_dict.".format(thread))
        if self.path != "":
            self.save(self.path)
        total_time = time.time()-training_start_time
        print("\nTraining time for {} epochs:".format(n_epochs),
              round(total_time,2), "\t per epoch:", round(total_time/n_epochs,2))
        # if multip:
        #     del process_group
        print("\nNumber of parameters trained:",self.n_params,end="\n")
        return self.best_score
    
    def train_loop(self,
                   data_set,
                   fraction:float=1, # fraction of training_data_used
                   n_parallel = 1,
                   thread = 0,
                   frequency = "m"):
        # import pdb; pdb.set_trace()
        train_props = data_set.get_props(frequency,"train") |{}
        self.size = max(1,int(train_props["n_batches"]*fraction))
        self.SIZE = train_props["len"]
        
        self.objective.clear()
        
        self.calc_batches = [int(self.size/4*value)-1 for value in range(1,5)]
        if self.verbose>0: self.timer("train_pre")
        batches = [*range(self.size)]
        batch_shift = int(thread/n_parallel*self.size)
        batches = [*batches[batch_shift:],*batches[:batch_shift]]
        # rd.shuffle(batches)
        self.calc_batches = [batches[i] for i in self.calc_batches]
        self.calc_batch_counter = 1
        for batch in batches:
            self.train_batch(data_set,frequency,batch)
            # self.vram_analysis.append(to.cuda.memory_allocated())
        # self.train_thread_manager(data_set, frequency)
        
        self.objective.result("train")
        if self.verbose>0: self.timer("train_result")
       # def train_thread_manager(self,data_set,frequency):    
       #     workers = []
       #     for worker in range(self.parallel_workers):
       #         thread = th.Thread(target=self.train_worker,
       #                            kwargs={"data_set":data_set,"frequency":frequency})
       #         workers.append(thread)
       #     for w in workers:
       #         w.start()
       #     for w in workers:
       #         w.join()
       #     print("Processes complete.")
       # def train_worker(self,data_set,frequency):
       #     while len(self.jobs)>0:
       #         batch = self.jobs.pop()
       #         self.train_batch(data_set,frequency,batch)
       
    def data_batch_preprocessing(self,X,y,meta, frequency, data_mode="numpy",split="train"):
        
        side = np.array([])
        if y is not None:
            if len(y.shape)>1:
                if y.shape[1]>1:
                    side = y[:,1:]
                    y = y[:,:1]
        
        if meta != {}:
            ## in case some preprocessing has to be done for meta
            pass
        
        if data_mode == "numpy":
            for frequency_ in X.keys():
                X[frequency_] = to.tensor(X[frequency_],device=self.device, dtype=to.float64)
                # X = X.share_memory_()
            for frequency_ in meta.keys():
                meta[frequency_] = to.tensor(meta[frequency_],device=self.device, dtype = to.float64)
        
        ## SEND TO TENSOR and change target for autoencoder network:
        
        if self.model_classification != "AE":
            ## send y and side to tensor if the datatype is numpy
            if data_mode == "numpy": 
                y = to.tensor(y,device=self.device, dtype=to.float64)
                side = to.tensor(side,device=self.device, dtype=to.float64)
                # X = to.tensor(X,device=self.device, dtype=to.float64)
        elif not self.instrumented: # meaning it is not instrumented AE
            ## if AE works with X*y as target
            if type(X[frequency])==np.ndarray: 
                y = to.tensor(X[frequency].copy(),device=self.device, dtype=to.float64)
                # y = to.tensor(X.copy(),device=self.device, dtype=to.float64)
            else:
                y = to.clone(X[frequency])
        else:
            if type(X[frequency])==np.ndarray:
                y = to.tensor(X[frequency].copy()*y,device=self.device, dtype=to.float64)
            else:
                for dim in range(len(X.shape)-len(y.shape)):
                    y = y.unsqueeze(1)
                y = to.clone(X[frequency]*y)
        if self.side == []:
            side = []
            
        return X, y, meta, side
    
    def add_vram(self,name):
        self.vram_analysis.append({name: to.cuda.memory_allocated()})
        pass
    
    def train_batch(self,data_set,frequency,batch):
        # import pdb; pdb.set_trace()
        
        ## get batches from dataset class:
            ## X: dict of {frequency:batch array or tensor}
            ## y: array or tensor of output
            ## meta: dict of {frequency: meta}
        if data_set.data_mode != "numpy":
            X, y, meta = data_set.get_train(
                frequency, batch, device = self.device)
        else:
            X, y, meta = data_set.get_train(
                frequency, batch, device = "cpu")
        if self.verbose>0: self.timer("train_get_data")
        
        self.add_vram("data_loading")
        X, y, meta, side = self.data_batch_preprocessing(
            X, y, meta, data_mode = data_set.data_mode, frequency = frequency,
            split= "train")
        
        self.add_vram("data_prprocessing")
        ## side is extracted from y. it could be seperated before...
        
        # shape 
        X_shape = X[frequency].shape
        
        ## error if X is empty
        if X_shape[0] == 0:
            raise ValueError("X shape:",{xx:X[xx].shape for xx in X.keys()},
                             "\nX type",type(X[frequency]),
                             "\nbatch",batch,"of",self.size)
            
            
        if not self.multifreq:
            X= X[frequency]
            if frequency in meta.keys():
                meta = meta[frequency]
                
        if self.verbose>0: self.timer("train_make_tensor")
        
        if X_shape[0]>1: # accounting for error in tensor flow if the batch is too small
            with to.cuda.stream(self.Stream):
                pred = self.model(X,objective = self.objective,meta=meta) # local prediction for batch
                self.add_vram("model forward")
                self.optimizer.zero_grad()
                if len(pred.shape)>len(y.shape):
                    pred = to.squeeze(pred) # squeeze pred if y has different number of axis
            if self.verbose>0: self.timer("train_model")
            
            ## compute local loss
            self.objective(pred, y, side, self.model, self.optimizer,self.Stream)
            pred = pred.detach()
            self.add_vram("model backward")
            if self.verbose>0: self.timer("train_objective")
        else:
            pred = None
        del X
        if batch in self.calc_batches:
            self.objective.int_result(batch= self.calc_batch_counter, 
                                      n_batches=len(self.calc_batches))
            self.calc_batch_counter+=1
        if self.verbose>0: self.timer("train_objective")
        self.Scheduler_Monitor.train(
            self.scheduler,
            self.objective.loss)
        self.add_vram("model scheduler and int_result")
        if self.verbose>0: self.timer("train_scheduler")
        to.cuda.empty_cache()
        del y, pred, side
    
    def val_loop(self,
                 data_set,
                 frequency):
        # import pdb; pdb.set_trace()
        val_props = data_set.get_props(frequency,"val") |{}
        size = val_props["n_batches"]
        self.SIZE = val_props["len"]
        
        self.objective.clear()
        
        if self.verbose>0: self.timer("val_pre")
        with to.no_grad():
            for batch in range(size):
                # batch = 0
                if data_set.data_mode != "numpy":
                    X, y, meta = data_set.get_val(
                        frequency,batch, device = self.device)
                else:
                    X, y, meta = data_set.get_val(
                        frequency,batch, device = "cpu")
                # X, y = data_set.get_val(frequency,self.side,batch)
                if self.verbose>0: self.timer("val_get_data")
                
                X, y, meta, side = self.data_batch_preprocessing(
                    X, y, meta, data_mode = data_set.data_mode, frequency = frequency,
                    split= "val")
                
                # side = np.array([])
                # if y is not None:
                #     if len(y.shape)>1:
                #         if y.shape[1]>1:
                #             side = y[:,1:]
                #             y = y[:,:1]
                # X = to.tensor(X,device=device, dtype=to.float64)
                # if self.model_type not in [AE_1d]:
                #     y = to.tensor(y,device=device)
                #     side = to.tensor(side,device=device)
                # elif not self.instrumented:
                #     y = X.clone()
                # else:
                #     y = to.tensor(y,device=device)
                #     for dim in range(len(X.shape)-len(y.shape)):
                #         y = y.unsqueeze(1)
                #     y = X*to.tensor(y)
                
                if self.verbose>0: self.timer("val_make_tensor")
                if X[frequency].shape[0]>1:
            
                    if not self.multifreq:
                        X= X[frequency]
                        if frequency in meta.keys():
                            meta = meta[frequency]
                    # with to.no_grad():
                    pred = self.model(X,objective = self.objective,meta = meta)
                    if self.verbose>0: self.timer("val_step")
                    if len(pred.shape)>len(y.shape):
                        pred = to.squeeze(pred)
                    self.objective(pred, y, side, self.model, optimizer=None)
                    del pred
                    if self.verbose>0: self.timer("val_objective")
                del X
                
        test_loss,test_penalty = self.objective.result("val")
        print()
        return test_loss, test_penalty
    
    def test(self,
             data_set, #dataset or PhD4_dataset
             quantiles = 10,
             limit = None,
             frequency = None):
        ## class model test
        print_hint2("Testing results for model",1,1,width_field = 50)
        self.get_model_type()
        # import pdb; pdb.set_trace()
        date_entity = data_set.date_entity[frequency]
        self.model.eval()
        size = data_set.get_props(frequency,"test")["n_batches"]
        SIZE = 0
        test = data_set.get_test_y(frequency,self.side)
        pred = []
        # Y    = []
        
        for batch in range(size):
            if data_set.data_mode != "numpy":
                X, y, meta = data_set.get_test(
                    frequency,batch, device = self.device)
            else:
                X, y, meta = data_set.get_test(
                    frequency,batch, device = "cpu")
            # X, y = data_set.get_val(frequency,self.side,batch)
            if self.verbose>0: self.timer("test_get_data")
            
            X, y, meta, side = self.data_batch_preprocessing(
                X, y, meta, data_mode = data_set.data_mode, frequency = frequency,
                split= "test")
            
            if self.verbose>0: self.timer("test_make_tensor")
            if X[frequency].shape[0]>0:
        
                if not self.multifreq:
                    X= X[frequency]
                    if frequency in meta.keys():
                        meta = meta[frequency]
                
                with to.no_grad():
                    SIZE += y.shape[0]
                    # Y.append(y)
                    pred.append(self.model(X,objective = self.objective,meta = meta).\
                                cpu().detach().numpy())
            del X
        # Y = np.concatenate(Y)
        pred = np.concatenate(pred)
        
        
        
        # if self.model_type not in [AE_1d]:
        #     test = copy.deepcopy(data_set.get_test_y(frequency,self.side))
        # else:
        #     test = copy.deepcopy(data_set.get_test_x(frequency))
        #     for batch in range(size):
        #         test = np.append(test,data_set.get_test_x(frequency,batch),axis=0)
        #     if self.instrumented:
        #         y = copy.deepcopy(data_set.get_test_y(frequency,self.side))
        #         for dim in range(len(test.shape)-len(y.shape)):
        #             y = np.expand_dims(y,1)
        #         test = test*y
        # with to.no_grad():
        #     for batch in range(size):
        #         # batch = 0
        #         X = data_set.get_test_x(frequency,batch)
        #         SIZE+=X.shape[0]
        #         X = to.tensor(X,device=device, dtype=to.float64)
        #         pred = np.append(pred,self.model(X,objective = self.objective,meta = meta).\
        #                          cpu().detach().numpy(),axis=0) #
        #     del X
            
        # import pdb; pdb.set_trace()
        test["pred"] = pred
        test.rename({data_set.target_variable:"test"},inplace=True,axis=1)
        
        if test["test"].isna().sum()!=0:
            print("WARNING in model.test: return series of frequency {} \
                  has NaN values for data preceeding{}".format(
                  data_set.target_variable,data_set.test_date))
            test = test[~test["test"].isna()]
        self.test_predictive, self.test_portfolio, test, self.pred_stats = \
            self.objective.test_result(
                test, 
                self.side,
                date_entity=date_entity, 
                target_periods=data_set.target_periods, 
                return_mode=data_set.return_mode,
                limit = limit)
        self.test_data = test[["pred","test",*self.side]]
        
        self.portfolio_monthly = pd.DataFrame()
        self.prediction_stats = {}
        
    def apply(self,
              data_set,
              date,
              frequency = None):
        self.model.eval()
        X, y = data_set.get_appl(frequency,date)
        X = to.tensor(X,device=self.device, dtype=to.float64)
        pred = self.model(X)
        pred = pred.cpu().detach().numpy() # self.objective.loss_fn.norm_pred(pred).cpu().detach().numpy()
        y["pred"] = pred
        y.sort_values(by="pred",inplace=True,ascending=False)
        return y
    
    def __del__(self):
        self.model.to("cpu")
        del self.model
    
    @property
    def n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


class TO_DDP_PG():
    def __init__(self,rank=0,masterport =None, evade = False):
        '''
        rank: id of processes
        world_size: number of processes in group
        '''
        ### multi_processing GPU: 
        # backend = "gloo"
        if masterport is None:
            masterport = 6818 # 12355
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(masterport)
        os.environ["NCCL_DEBUG"] = "INFO"
        if not to.distributed.is_initialized():
            # try:
            print("Initialising Process Group...")
            init_process_group(backend=backend,
                               rank=rank,
                               timeout = dt.timedelta(seconds=300),
                               world_size=len(devices)
                               )
            print("Process group initialised.")
            # except:
                # if backend == "nccl" and evade:
                #     print("WARNING: Error initialising process group with backend = 'nccl'.")
                #     print("\t\t Retrying with 'gloo'...")
                #     init_process_group(backend="gloo",
                #                         rank=rank,
                #                         timeout = dt.timedelta(seconds=300),
                #                         world_size=len(devices)
                #                         )
                #     print("Process group initialised.")
                # elif not evade:
                #     print("init_process_group failed")
                # else:
                #     raise
        # to.cuda.set_device(rank)
        # self.iterator = iterator
    def __del__(self):
        # if to.distributed.is_initialized():
        #     destroy_process_group(
        #         )
        pass
    
    
def PhD2_year_job_thread(data_set,model_type,jobs,year_results,thread_nr,
                         timers,train_validate_pars,test_pars,
                         verbose = 0, model_file_path = ""):
    
    while len(jobs)>0:
        job_name, model_parameters = jobs.pop()
        if "verbose" not in model_parameters.keys():
            model_parameters["verbose"] = verbose
        model_job = model(model_type,
                          **model_parameters)
        model_job.objective.thread = thread_nr
        model_job.train_validate(
            data_set,
            thread = thread_nr,
            **train_validate_pars)
        model_job.test(
            data_set,
            **test_pars)
    
        if year_results["portfolio"].shape[0]==0:
            year_results["portfolio"] = model_job.test_data.rename({"pred":job_name},axis=1)    
        else:
            year_results["portfolio"][job_name] = model_job.test_data["pred"]
        year_results["prediction_results"][job_name] = model_job.test_predictive
        model_job.test_portfolio["model"] = job_name
        year_results["portfolio_results"].append(model_job.test_portfolio)
        model_job.pred_stats["model"] = job_name
        year_results["prediction_stats"].append(model_job.pred_stats)
        timers[job_name] = model_job.timer.time
        if model_file_path != "":
            model_job.save(model_file_path+f"{job_name}.pkl")

def fc_naive(portfolio,application, objective, 
             args,
             **kwargs):
    columns = portfolio.columns.tolist()
    columns.remove("test")
    if "side" in args.keys():
        for side in args["side"]:
            columns.remove(side)
    portfolio["pred"] = portfolio[columns].mean(axis=1)
    appl_ret = pd.DataFrame()
    if application.shape[0]!=0:
        application["pred"] = application[columns].mean(axis=1)
        appl_ret = objective.loss_fn.norm_pred(application["pred"])
    return objective.test_result(portfolio,**args),\
        appl_ret
            
################################################################################
################################################################################
################################  Dataset Class  ###############################
################################################################################
################################################################################


class PhD4_dataset():
    '''
    Renovation of dataset class with dataloader and memory management.

    '''
    date_entities = {"date","day","month"}
    label_file = "labels{frequency:s}.csv"
    map_file = "maps{frequency:s}.csv"
    data_identification_file = "ident{frequency:s}.csv"
    x_name = "data_b{batch:d}x{frequency:s}.h5"
    y_name = "data_b{batch:d}y1.h5"
    pointer_columns = ["pointer_m_q","quarter_history","dist_last_quarter",
                       "month_history"]
    reshape_ = {2:[1,0],
                3:[2,1,0],
                4:[3,2,1,0]
               }
    hvf_able = ["d","m"] # frequencies that can be high volume
    date_entinty_defaults = {
        "m":"month",
        "d":"day",
        "q":"date"}
    def __init__(self,
                 file_path:str,#file path
                 test_len:float | int=12, #float between 0 and 1 or int
                 val_len:float | int=24, # float between 0 and 1 or int
                 load_len:float | int = 252, # flaot between 0 and 1 or int
                 appl: bool = False, # float between 0 and 1 or int
                 test_date:str = "2022-12-31", # can be fixed date of test end
                 
                 input_frequencies = ["m","q"],
                 target_variable:str = "ret",
                 use_columns:list=[],
                 meta_series:dict = {},
                 merge_freq = True,
                 ind = False, #wether intdustry adjustments have been computed for all columns
                 
                 mode_validation = "rcv", # one of ["vw","tocv","rcv"]
                 data_mode:str = "numpy",
                 batch_size:int = 256,
                 return_mode:str = "log",
                 obj_series_add = {"input":[],
                                   "class":None},
                 date_format = "%Y-%m-%d",
                 preprocessing = {"batchwise":{},
                                  "dataset":{},
                                  "filter":{}
                                  },
                 lookback:dict = {"m":[0,0],"q":0}, #dict of lists
                 batch_factor:int = 1,
                 identifier="tick",
                 verbose=0,
                 main = None, ## can define the leading frequency as a different one from the first in input_frequencies
                 **kwargs):
        '''
        
        '''
        # import pdb; pdb.set_trace()
        _error_base_ = "Class PhD4_dataset.__init__: " # error base string displayed in all local erros
        
        ##################################################
        #  Setting object constants with default values  #
        ##################################################
        
        self.merged = False ## has the dataset been merged yet?
        self.lookback_min = {} ## minimum lookback available data to not drop from dataset
        self.reshape = copy.deepcopy(self.reshape_) # reshape of data into pytorch module structure
        self.dim = [] # dimension of training dataset, to be determined during loading
        self.y_index_columns = [] # columns to use for indexation in the dataset
        self.locked = 0 # parallel processing lock
        self.batched = {i:False for i in input_frequencies} ## wether the series is in batches
        self.split = False ## wether the dataset is split
        self.dfed = True
        self.column_names = {}
        self.shuffle_iteration = 0
        self.logged_models = {}
        
        ################################################################################
        ########  Setting object constants from object construction parameters  ########
        ################################################################################
        
        self.file_path = file_path # path to the files
        if test_date is None: 
            test_date = dt.datetime.today()
        else:
            self.test_date = test_date ## last date for testing period.
        self.appl = appl
        
        
        self.input_frequencies = input_frequencies+[] ## frequencies of input data
        self.meta_series = meta_series
        if data_mode not in ["numpy","torch"]:
            raise ValueError(_error_base_+f"Illegal datamode. Got handed '{str(data_mode):s}'"+\
                             "\n Legal values: ['numpy','torch']")
        self.data_mode = data_mode ## data mode for handling the data: either 'numpy' or 'torch'
        
        self.merge_freq   = merge_freq # number of storage frequencies for training
        self.ind = ind ## does the dataset contain universal industry variables?
        
        self.mode_validation = mode_validation # mode of validation
        self.batch_size = batch_size # size of each training batch
        self.return_mode = return_mode # wether to use simple or log returns
        self.date_format = date_format
        self.preprocessing = preprocessing # list of preprocessing objects
        self.lookback = lookback ## lookback setup for all frequencies
        self.identifier = identifier ## identifier name used in data
        self.verbose = verbose # mode of status report, active > 0
       
        ## minimum values for allowing lookback observations are an optional kwarg
        if "lookback_min" in kwargs.keys():
            self.lookback_min.update(kwargs["lookback_min"])
        
        
        ################################################################################
        #########  Setting object constants from calculation and verification  #########
        ################################################################################
        
        ## type and value check of input lengths
        PhD4_dataset_len_error = \
            f"\ntype(test_len):{type(test_len):}, value: {test_len:}"+\
            f"\ntype(val_len):{type(val_len):}, value: {val_len:}"+\
            f"\ntype(load_len):{type(load_len):}, value: {load_len:}"
        if len(set([type(test_len),type(val_len)])) != 1:
            raise TypeError(_error_base_+\
                "\ntest_len and val_len must be of same type."+PhD4_dataset_len_error)
        elif type(test_len) is float:
            if test_len+val_len>1:
                raise ValueError(_error_base_+\
                    "\ntest_len and val_len together can not exceed 1 if they are float!")
        elif type(test_len) is int:
            if test_len +val_len>load_len:
                raise ValueError(_error_base_+\
                    "\ntest_len and val_ken cannot exceed load_len if they are int!"+\
                        PhD4_dataset_len_error)
        
        self.main = main
        if self.main is None:
            self.main = input_frequencies[0] ## main frequency, target frequency
        elif self.main not in input_frequencies:
            # raise errpr if main not in input_frequencies
            raise ValueError(_error_base_+ f"illegal main. Found type(main):{type(main):}, value: {main:}"+\
                f"\nbut main needs to be in input_frequencies: {input_frequencies:}.")
        self.meta_list = [*meta_series.values()]
        #setting paramters for import
        
        ## check validity of validation_mode
        if mode_validation not in ["rcv","tocv","vw"]:
            raise ValueError(_error_base_+ f"Illegal mode_validation. Found: {mode_validation:}"+\
                             'Must be in ["rcv","tocv","vw"]')
        
        ## check type and validity of test_date and date format
        if type(date_format) == str and type(test_date) == dt.datetime:
            test_date = test_date.strptime(date_format)
        elif type(date_format) == dt.datetime and type(test_date) == str:
            test_date = dt.datetime.strftime(test_date)
        else:
            if type(date_format)!= type(test_date):
                raise TypeError("Class: PhD4_dataset.__init__: Illegal type for date_format or test_date:"+\
                                f"\ntype(date_format):{type(date_format):}, value: {date_format:}"+\
                                f"\ntype(test_date):{type(test_date):}, value: {test_date:}")
        
        ## check return mode validity
        if return_mode not in ["log","simple"]:
            raise ValueError(_error_base_+ f"Illegal return mode. Found: {return_mode:}"+\
                             'Must be in ["log","simple"]')
                
        ## check if preprocessing is complete:
        for preprocessing_point in ["batchwise","dataset","filter"]:
            if preprocessing_point not in self.preprocessing.keys():
                self.preprocessing[preprocessing_point] = {}
            for frequency in self.input_frequencies:
                if frequency not in self.preprocessing[preprocessing_point].keys():
                    self.preprocessing[preprocessing_point][frequency] = []
        
        ## filling lookback_min with values, edges
        for frequency in self.input_frequencies:
            if frequency not in self.lookback.keys():
                self.lookback[frequency] = 1
        for key in lookback.keys():
            # key = "w"
            # import pdb; pdb.set_trace();
            if type(lookback[key]) == int:
                self.lookback[key] = [lookback[key],1,1]
            elif len(lookback[key]) < 3:
                self.lookback[key] = [*lookback[key],*[1]*(3-len(lookback[key]))]
            self.lookback[key][2] = lookback[key][0]//lookback[key][1]
            
            if key in self.lookback_min.keys():
                self.lookback_min[key] = min(
                    max(0,self.lookback_min[key]), 
                    self.lookback[key][0])
            else:
                self.lookback_min[key] = int((self.lookback[key][0]-1)/self.lookback[key][1])+1
            self.lookback[key][1] = max(1,self.lookback[key][1])
            
        self.set_target(target_variable) ## target variable used from y_name file
        self.batch_size_testval=(self.batch_size)*batch_factor # size of each validation or testing set
        
        ################################################################################
        ##################################  Metadata  ##################################
        ################################################################################
        
        # start of loading process, start with metadata
        # self.nhvf   = list(set(self.input_frequencies).difference(set(self.hvf_able))) # other series
        self.store_index = {}
        self.store_x = {} ## data storage x
        self.store_y = {} ## data storage y
        self.props = {} ## properties
        self.ident = {} ## ident files
        self.labels = {} ## input columns of each frequency
        self.preserved = {}
        self.date_entity = {}
        self.maps = {}
        self.side = []
        self.lookback_activated = False
        self.split_ = False
                
        for frequency in self.input_frequencies:  
            # frequency = "w"
            self.labels[frequency]            = pd.read_csv(
                self.file_path+self.label_file.format(frequency=frequency),sep=";",index_col=0)
            self.maps[frequency]              = pd.read_csv(
                self.file_path+self.map_file.format(frequency=frequency), sep=";", index_col=0) 
            self.ident[frequency]             = {}
            self.ident[frequency]["train"]    = pd.read_csv(
                self.file_path+self.data_identification_file.format(frequency=frequency),
                sep=";",index_col=0)
            self.ident[frequency]["train"]    = self.ident[frequency]["train"].\
                sort_values("input_date").reset_index(drop=True)
            
        
        shift_unit, pars = {
            "w":[month_shift2,{"end":True}],
            "m":[month_shift2,{"end":True}],
            "d":[month_shift2,{"end":True}]}\
            [self.main]
        
        if self.appl:
            self.appl_date = shift_unit(self.test_date,shift = 12,**pars)
        else:
            self.appl_date = shift_unit(self.test_date,shift = 0,**pars)
        self.val_date = shift_unit(self.test_date,shift=-test_len,**pars)
        self.train_date = shift_unit(self.val_date,shift=-val_len,**pars)
        self.train_start = shift_unit(self.test_date,shift=-load_len,**pars)
        del shift_unit, pars
        
        timer = timeit()
        
        #### Minimum date for import
        '''
        Shifting train_start date further back depending on frequency.
        
        daily:      lookback +2 int divided by 21 +1
        weekly:     lookback +2 int divided by 4 +1
        monthly     lookback +1
        quarterly:  lookback *3 +1
        '''
        
        self.import_min_date = {frequency:self.train_start+"" for frequency in self.input_frequencies}
        if "w" in self.input_frequencies:
            self.import_min_date["w"] = month_shift2(
                self.import_min_date["w"], 
                -((self.lookback["w"][2]+2)//4+1)
                )
        if "m" in self.input_frequencies:
            self.import_min_date["m"] = month_shift2(
                self.import_min_date["m"], 
                -(self.lookback["m"][2]+1)
                )
        if "q" in self.input_frequencies:
            self.import_min_date["q"] = month_shift2(
                self.import_min_date["q"], 
                -(int(self.lookback["q"][2]*3)+1)
                )
        if "d" in self.input_frequencies:
            self.import_min_date["d"] = month_shift2(
                self.import_min_date["d"], 
                -((self.lookback["d"][2]+2)//21+1)
                )
        
        ################################################################################
        ########################  Data Import and Preprocessing  #######################
        ################################################################################
        
        for frequency in self.input_frequencies:
            # frequency = "w"
            
            self.ident[frequency]["train"] = self.ident[frequency]["train"]\
                [(self.ident[frequency]["train"]["target_date"]<=self.appl_date)&\
                 (self.ident[frequency]["train"]["target_date"]>self.import_min_date[frequency])].\
                    reset_index(drop=True)
            self.store_x[frequency] = {}
            self.store_index[frequency] = {}
            self.store_x[frequency]["train"] = []
            if frequency == self.main:
                self.store_y["train"] = []
            for batch in self.ident[frequency]["train"]["batch"]:
                self.store_x[frequency]["train"].append(
                    pd.read_hdf(self.file_path+self.x_name.\
                           format(batch= batch,
                                  frequency = frequency),
                           "data"))
                if frequency == self.main:
                    self.store_y["train"].append(
                        pd.read_hdf(self.file_path+self.y_name.\
                                    format(batch= batch),
                                    "data"))
            self.store_x[frequency]["train"] = pd.concat(self.store_x[frequency]["train"])
            
            if len(use_columns)!= 0:
                # import pdb; pdb.set_trace()
                local_cols = list(set(use_columns).intersection(
                    self.store_x[frequency]["train"].columns.tolist()))
                self.store_x[frequency]["train"] = self.store_x[frequency]["train"][local_cols]
            
            self.column_names[frequency] = self.store_x[frequency]["train"].columns.tolist()
            self.date_entity[frequency] = [i for i in self.store_x[frequency]["train"].index.names\
                                     if i != self.identifier][0]
            if frequency == self.main:
                self.y_index_columns = self.store_y["train"][0].index.names
                self.store_y["train"] = pd.concat(self.store_y["train"]).\
                    reset_index()
            # import pdb; pdb.set_trace()
            date_entity = self.date_entity[frequency] + ""
            self.props[frequency] = {}
            
            print("{}: loading complete.".format(frequency), timer())
            
            ################################################################################
            ################################  Preprocessing  ###############################
            ################################################################################
            
            #### creating objectives from input data: ##lagging happens through class 
            if frequency == self.main and len(obj_series_add["input"])>0:
                
                self.store_y["train"] = pd.merge(
                    self.store_y["train"],
                    self.store_x[frequency]["train"][obj_series_add["input"]].fillna(0),
                    left_on = self.store_x[frequency]["train"].index.names,
                    right_index=True)
                self.side.extend(obj_series_add["input"])
                self.store_y["train"].set_index(
                    self.store_x[frequency]["train"].index.names,inplace=True)
                if obj_series_add["class"] is not None:
                    self.store_y["train"] = obj_series_add["class"](
                        self.store_y["train"],date_entity)
                # for column in obj_series_add["input"]:
                #     del self.store_y["train"][column]
                
                if frequency == self.main:
                    self.store_y["train"] = self.store_y["train"].sort_index()#.reset_index()
                print("{}: secondary objective complete.".format(frequency), timer())
            elif frequency == self.main: 
                self.store_y["train"].set_index(
                    self.store_x[frequency]["train"].index.names,inplace=True)
                self.store_y["train"] = self.store_y["train"].sort_index()
                print("{}: no secondary objective.".format(frequency))
            
            self.store_x[frequency]["train"].sort_index(inplace=True)
            
            if frequency == self.main:
                # import pdb; pdb.set_trace()
                indices = self.store_y["train"][~self.store_y["train"][self.target_variable].isna()].index
                self.store_y["train"] = self.store_y["train"].loc[indices,:]
                self.store_x[frequency]["train"] = self.store_x[frequency]["train"].loc[indices,:]
            #### batch preprocessing.
            if "batchwise" in preprocessing.keys():
                self.store_x[frequency]["train"] = self._preprocess_batch(
                    self.store_x[frequency]["train"],
                    frequency=frequency)
                print("{}: batch preprocessing complete.".format(frequency), timer())
            ## just inc case some data is dropped during preprocessing:
            if frequency == self.main:
                keep_indices = self.store_x[frequency]["train"].index
                self.store_y["train"] = self.store_y["train"].loc[keep_indices,:]
                self.store_y["train"] = self.store_y["train"].sort_index()
                self.store_y["train_backup"] = self.store_y["train"].copy()
                
            self.store_x[frequency]["train"] = self.store_x[frequency]["train"].sort_index()
            self.store_index[frequency]["train"] = self.store_x[frequency]["train"].index
            
        ################################################################################
        ###########################  Mergeing and Splitting  ###########################
        ################################################################################
        '''
        Reorder:
            0. Dataset preprocessing
            1. Lookback
            2. Mergeing
            3. Splitting
        '''
        # import pdb; pdb.set_trace()
        # train_start = self.train_start+""
        # self.train_start = "1800-01-01"
        self.splitter()
        self.preprocessing_dataset(frequency,timer,data_mode_function = "numpy")
        self.unsplitter(False)
        # import pdb; pdb.set_trace()
        # self.train_start = train_start+""
        if self.merge_freq:
            self._synchronise_frequencies(True)
            print("{}: frequency merging complete complete.".format(frequency), timer())
        if self.lookback[self.main][0] >1:
              self.lookback_build()
              print("{}: lookback build complete complete.".format(frequency), timer())   
        elif not self.merge_freq:
            self._synchronise_frequencies(False)
        ## splitting into train, val, test, and appl. Also takes care of batching
        self.splitter()
        self.time = [0,0,0,0]
        to.cuda.empty_cache()
    
    def model_log(self,thread):
        self.logged_models[thread] = 0
    
    def model_log_add(self,thread):
        self.logged_models[thread] += 1
    
    def model_get_status(self):
        return min([log for thread,log in self.logged_models.items()])
   
    def splitter(self,rebatch = True,train_val_split={}):
        ## splitting into train, val, test
        # if self.lookback_activated: import pdb; pdb.set_trace()
        self.undfer()
        self.unbatcher()
        if self.split_:
            return False
        date_entity = self.date_entity[self.main]
        date_ranges = [self.appl_date, self.test_date, self.val_date, self.train_date,
                       self.train_start]
        if type(self.store_index[self.main]["train"].get_level_values(
                date_entity).values[0]) is dt.date:
            date_ranges = [dt.datetime.strptime(date,"%Y-%m-%d").date() for date in date_ranges] 
        
        split_date_index = 0
        for split in ["appl","test","val","train"]:
            # split = "appl"
            for frequency in self.input_frequencies:
                # frequency = "m"
                
                local_index = pd.DataFrame(
                    np.arange(len(self.store_index[frequency]["train"])),
                    index=self.store_index[frequency]["train"], columns = ["index"])
                date_entity = [name for name in local_index.index.names if name != self.identifier][0]
                if split in ["train","val"] and train_val_split != {}:
                    select_indices = local_index.loc[
                        local_index.index.get_level_values(date_entity).isin(train_val_split[split]),
                        "index"].values
                else:
                    select_indices = local_index.loc[
                        (local_index.index.get_level_values(date_entity) <= date_ranges[split_date_index])&\
                        (local_index.index.get_level_values(date_entity) > date_ranges[split_date_index+1]),
                        "index"].values
                self.store_index[frequency][split] = self.store_index[frequency]["train"][
                        select_indices]
                self.store_x[frequency][split] = self.store_x[frequency]["train"][
                        select_indices,...]
                if frequency == self.main:
                    self.store_y[split] = self.store_y["train"][select_indices,...]
                    index_names = [self.identifier,self.date_entity[frequency]]
                    self.store_y[f"{split:}_df"] = self.store_y["train_backup"].reset_index().copy()
                    self.store_y[f"{split:}_df"] = self.store_y[f"{split:}_df"].loc[
                        select_indices,:].set_index(index_names)
                _batch_size = self.batch_size
                if split != "train": _batch_size = self.batch_size_testval
                
                if len(self.store_x[frequency][split].shape) <3:
                    self.store_x[frequency][split] = self.store_x[frequency][split][...,np.newaxis]
                self.props[frequency][split] = {
                    "dim":self.store_x[frequency][split].shape[1:],
                    "batch":0,
                    "len":self.store_x[frequency][split].shape[0],
                    "n_batches": (self.store_x[frequency][split].shape[0]-1)//\
                        _batch_size+1}
                
            split_date_index += 1
        self.split_ = True
        if rebatch:
            self.batcher()
            
    def unsplitter(self,rebatch = True):
        ## reverse splitting from train, val, test
        # import pdb; pdb.set_trace()
        self.undfer()
        self.unbatcher()
        if not self.split_:
            return False
        ## delete val, test, and appl files
        existing_splits = list(self.store_x[self.main].keys())
        existing_splits.remove("train")
        for frequency in self.input_frequencies:
            # frequency = "w"
            self.store_x[frequency]["train"] = [self.store_x[frequency]["train"]]
            # self.store_index[frequency]["train"] = [self.store_index[frequency]["train"]]
            if frequency == self.main:
                self.store_y["train"] = [self.store_y["train"]]
            for _split in existing_splits:
                # _split = "appl"
                self.store_x[frequency]["train"].append(self.store_x[frequency][_split])
                del self.store_x[frequency][_split]
                self.store_index[frequency]["train"] = \
                    self.store_index[frequency]["train"].append(self.store_index[frequency][_split])
                del self.store_index[frequency][_split]
                del self.props[frequency][_split]
                if frequency == self.main:
                    self.store_y["train"].append(self.store_y[_split])
                    del self.store_y[_split]
            self.store_x[frequency]["train"] = np.concatenate(self.store_x[frequency]["train"])
            if self.store_x[frequency]["train"].shape[-1] == 1:
                self.store_x[frequency]["train"] = np.squeeze(self.store_x[frequency]["train"])
            # self.store_index[frequency]["train"] = pd.concat(self.store_index[frequency]["train"])
            if frequency== self.main:
                self.store_y["train"] = np.concatenate(self.store_y["train"])
                self.store_y["train_backup"] = self.store_y["train_backup"].loc[
                    self.store_index[frequency]["train"],:]
        self.split_ = False
        if rebatch:
            self.batcher()
   
    def undfer(self,frequency: str | list = None):
        
        if self.dfed == False:
            return
        frequencies = [i for i,j in self.batched.items() if j==False]
        if frequency is not None:
            if type(frequency) != list:
                frequencies = [frequency]
        splits = list(self.store_x[self.main].keys())
        
        ## error handling
        if set(frequencies).difference(set(self.input_frequencies)) != set():
            raise ValueError(f"Classe PhD4_dataset.undfer: Unknown frequencies in input. Got {frequencies:}, "+\
                             f"expected a subset of {self.input_frequencies:}!")
                
        for frequency in frequencies:
            self.column_names[frequency] = self.store_x[frequency][splits[0]].columns.to_list()
        
        for split in splits:
            # split = "train"
            for frequency in frequencies:
                if type(self.store_x[frequency][split]) is pd.DataFrame:
                    self.store_index[frequency][split] = self.store_x[frequency][split].index
                    self.store_x[frequency][split] = self.store_x[frequency][split].to_numpy()
            if self.main in frequencies:
                if type(self.store_y[split]) is pd.DataFrame:
                    self.store_y[f"{split:}_backup"] = self.store_y[split].copy()
                    self.store_y[split] = self.store_y[split][[
                        self.target_variable,*self.side,*self.meta_list]].\
                        to_numpy()
        
        self.dfed = False
    
    def dfer(self,frequency: str | list= None):
        
        # import pdb; pdb.set_trace()
        if self.dfed == True:
            return
        elif self.lookback_activated:
            print("df is not a valid datatype if lookback is activated.")
            return
        
        ## only batched dataset can be dfed
        frequencies = [i for i,j in self.batched.items() if j==False]
        if frequency is not None:
            if type(frequency) != list:
                frequencies = [frequency]
        
        ## error handling
        if set(frequencies).difference(set(self.input_frequencies)) != set():
            raise ValueError(f"Classe PhD4_dataset.dfer: Unknown frequencies in input. Got {frequencies:}, "+\
                             f"expected a subset of {self.input_frequencies:}!")
        self.unbatcher(frequencies)
                
        splits = self.store_x[self.main].keys()
        
        for frequency in frequencies:
            column_names = self.column_names[frequency]
            for split in splits:
                self.store_x[frequency][split] = \
                    pd.DataFrame(self.store_x[frequency][split],columns = column_names,
                                 index=self.store_index[frequency][split])
                if frequency == self.main:
                    self.store_y[split] = self.store_y[f"{split:s}_backup"].copy()
        
        self.dfed = True
    
    def find_obs(self, index):
        '''
        This function finds a specific observation based on an index.
        
        !!! 
        First it finds the split that the datapoint is in.
        Then it may have to find the batch.
        It may also have to find the pointers
        '''
        split = None
        ## find split
        
        ## return obs
        if self.dfed:
            # easiest one
            return self.store_x[self.main][split].loc[index,:], \
                self.store_y[split].loc[index,:]
        else:
            pass
        
    def batcher(self,frequency: str | list = None,data_mode_function = None):
        ## default data_mode
        data_mode_local = self.data_mode+""
        if data_mode_function is not None:
            data_mode_local = data_mode_function+""
        
        ## default frequencies
        frequencies = [i for i,j in self.batched.items() if j==False]
        if frequency is not None:
            if type(frequency) != list:
                frequencies = [frequency]
        
        ## default splits
        splits = self.store_x[self.main].keys()
        
        ## error handling
        if set(frequencies).difference(set(self.input_frequencies)) != set():
            raise ValueError(f"Classe PhD4_dataset.batcher: Unknown frequencies in input. Got {frequencies:}, "+\
                             f"expected a subset of {self.input_frequencies:}!")
            
        ## convert to np.ndarray if type is pd.DataFrame
        self.undfer(frequencies)
        
        for split in splits:
            _batch_size = self.batch_size
            if split != "train": _batch_size = self.batch_size_testval
            if data_mode_local == "torch":
                ## convert into little torch chunks
                for frequency in frequencies:
                    self.store_x[frequency][split] = {device:list(to.split(to.tensor(
                        self.store_x[frequency][split],device=device, dtype=to.float64),
                        split_size_or_sections=_batch_size))
                            for device in devices}
                    if frequency == self.main:
                        self.store_y[split] = {device:list(to.split(to.tensor(
                            self.store_y[split],
                            device=device, dtype=to.float64),
                            split_size_or_sections=_batch_size))
                                for device in devices}
            else: 
                ## convert into little numpy chunks
                for frequency in frequencies:
                    # frequency = "w"
                    self.store_x[frequency][split] = {"cpu":list(np.split(
                        self.store_x[frequency][split],
                        indices_or_sections = [
                            i*_batch_size for i in \
                                range(1,(self.store_x[frequency][split].shape[0]-1)//_batch_size+1)]))
                            }
                    if frequency == self.main:
                        self.store_y[split] = {"cpu":list(np.split(
                            self.store_y[split],
                            indices_or_sections =[
                                i*_batch_size for i in \
                                    range(1,(self.store_y[split].shape[0]-1)//_batch_size+1)]))}
        for frequency in frequencies:
            self.batched[frequency] = True
    
    def unbatcher(self,frequency: str | list = None,data_mode_function = None):
        # import pdb; pdb.set_trace()
        ## default data mode
        data_mode_local = self.data_mode+""
        if data_mode_function is not None:
            data_mode_local = data_mode_function+""
        
        ## default frequencies
        frequencies = [i for i,j in self.batched.items() if j==True]
        if frequency is not None:
            if type(frequency) != list:
                frequencies = [frequency]
                
        ## default splits
        splits = self.store_x[self.main].keys()
        
        if set(frequencies).difference(set(self.input_frequencies)) != set():
            raise ValueError(f"Classe PhD4_dataset.batcher: Unknown frequencies in input. Got {frequencies:}, "+\
                             f"expected a subset of {self.input_frequencies:}!")
               
        for split in splits:
            if data_mode_local == "torch":
                for frequency in frequencies:
                    self.store_x[frequency][split] = to.cat(
                        self.store_x[frequency][split][devices[0]],dim = 0).\
                        cpu().detach().numpy()
                    if frequency == self.main:
                        self.store_y[split] = to.cat(
                            self.store_y[split][devices[0]],dim=0).\
                            cpu().detach().numpy()
            else:
                for frequency in frequencies:
                    self.store_x[frequency][split] = np.concatenate(
                        self.store_x[frequency][split]["cpu"], axis = 0)
                    if frequency == self.main:
                        self.store_y[split] = np.concatenate(
                            self.store_y[split]["cpu"], axis = 0)
        
        for frequency in frequencies:
            self.batched[frequency] = False
            
    def set_target(self,target_variable:str,target_periods:int = None):
        '''
        This method sets a new target variable. It can automatically extract the frequency
            from the name of the target variable if no target frequency is provided.

        Parameters
        ----------
        target_variable : str 
            Name of new target variable as used in the dataset.
        target_periods : int, DEFAULT None
            Frequency of the new target, used for performance evaluation.
        '''
        self.target_variable = target_variable
        if target_periods is None:
            print("Attention: Number of target periods automatically extracted!")
            target_periods_int = ""
            for letter in target_variable:
                if isnumber(letter):
                    target_periods_int+=letter
            if len(target_periods_int)>0:
                self.target_periods = int(target_periods_int)
            else:
                self.target_periods = 1
        else:
            self.target_periods = target_periods
        print(f"PhD4_dataset: Switched target variable! New target variable is: {target_variable:s}."+\
              f"\nThe frequency of this variable is {self.target_periods:d} {self.main:s}")
    
    def preprocessing_dataset(self,frequency,timer=timeit(),data_mode_function = None):
        # import pdb; pdb.set_trace()
        for process in self.preprocessing["dataset"][frequency]:
            if not process.trained:
                process.train(self,frequency=frequency)
            for split in self.store_x[frequency].keys():
                for device in self.store_x[frequency][split].keys():
                    for batch in range(len(self.store_x[frequency][split][device])):
                        self.store_x[frequency][split][device][batch] = \
                            process(self.store_x[frequency][split][device][batch])
        print("{}: autoencoder complete.".format(frequency), timer())
        # self.batcher(frequency,data_mode_function = data_mode_function)
    
    def _synchronise_frequencies(self,merge=False):
        # import pdb; pdb.set_trace()
        self.unsplitter(rebatch = False)
        self.dfer()
        ## synchronise values across different frequencies
        for frequency in self.input_frequencies:
            # frequency = "w"
            ## sort data
            sort_order = [self.identifier,self.date_entity[frequency]]
            self.store_x[frequency]["train"].sort_index(level = sort_order,inplace=True)
            self.store_index[frequency]["train"] =\
                self.store_index[frequency]["train"].sortlevel(level = sort_order)[0]
            self.store_index[frequency]["train"] = pd.DataFrame(index=self.store_index[frequency]["train"]).\
                reorder_levels(sort_order).index
            self.store_x[frequency]["train"] = self.store_x[frequency]["train"].to_numpy()
            if frequency == self.main:
                self.store_y["train"].sort_index(level = sort_order,inplace=True)
                self.store_y["train_backup"] = self.store_y["train"].copy()
                self.store_y["train"] = self.store_y["train"][[
                    self.target_variable,*self.side,*self.meta_list]].to_numpy()
            
        pointer_frequencies = self.input_frequencies+[]
        pointer_frequencies.remove(self.main)
        keep_indices = np.arange(self.store_y["train_backup"].shape[0])
        reference_indices = {frequency:None for frequency in pointer_frequencies}
        drop_indices = []
        for frequency in pointer_frequencies:
            # frequency = "m"
            pointer_column = f"{frequency:}_date"
            
            ## load the pointer table to join with target frequency
            reference_indices[frequency] = self.store_y["train_backup"][[pointer_column]].reset_index()
            reference_indices[frequency].set_index([self.identifier,pointer_column],inplace=True)
            
            ## load the pointed table of the frequency
            index_pointer = pd.DataFrame(data =0,
                index=self.store_index[frequency]["train"],columns = ["index"])
            index_pointer.index.names = [self.identifier,pointer_column]
            index_pointer["index"] = np.arange(index_pointer.shape[0])
            reference_indices[frequency] = pd.merge(
                reference_indices[frequency],index_pointer,
                # left_on = [self.identifier,pointer_column], right_on = [self.identifier,"date"],
                left_index=True, right_index=True,
                how="left")
            reference_indices[frequency].reset_index(inplace=True)
            drop_indices.extend(reference_indices[frequency][reference_indices[frequency]["index"].isna()].index)
            ## some observations have to be dropped because there are not enough lookback observations to back it up.
        keep_indices = list(set(keep_indices).difference(set(drop_indices)))
        
        ## drop observations unmatched because of insufficent lookback from self.main
        self.store_x[self.main]["train"] = [self.store_x[self.main]["train"]\
            [keep_indices,...]]
        self.store_y["train"] = self.store_y["train"][keep_indices,...]
        self.store_index[self.main]["train"] = \
            self.store_index[self.main]["train"][keep_indices]
        try:
            # self.store_y["train_backup"] = self.store_y["train_backup"].loc\
            #     [self.store_index[self.main]["train"],:]
            self.store_y["train_backup"] = pd.merge(
                self.store_y["train_backup"], pd.DataFrame(None,self.store_index[self.main]["train"]),
                left_index = True, right_index=True,how="inner")
        except:
            import pdb; pdb.set_trace();
        ## append non-main frequencies to main frequency.
        for frequency in pointer_frequencies:
            # frequency = "q"
            indices = reference_indices[frequency].loc[keep_indices,"index"].astype(int).tolist()
            self.store_x[frequency]["train"] = self.store_x[frequency]["train"][indices,...]
            if merge:
                self.store_x[self.main]["train"].append(self.store_x[frequency]["train"])
                self.column_names[self.main].extend(self.column_names[frequency])
                del self.store_index[frequency], self.column_names[frequency],self.store_x[frequency]
                self.input_frequencies.remove(frequency)
            else:
                self.store_index[frequency]["train"] =  self.store_index[self.main]["train"].copy()# [indices]
        if merge:
            self.store_x[self.main]["train"] = np.concatenate(self.store_x[self.main]["train"],axis=1)
            self.merged = True
            self.batched = {self.main:self.batched[self.main]}
        else:
            self.store_x[self.main]["train"] = self.store_x[self.main]["train"][0]
        self.dfed = False
    
    def _preprocess_batch(self,x,frequency):
        if frequency in self.preprocessing["batchwise"].keys():
            # if frequency == "q":import pdb; pdb.set_trace();
            for process in self.preprocessing["batchwise"][frequency]:
                # process = self.preprocessing["batchwise"][frequency][1]
                x = process(x)
        return x
    
    def _preprocess_data(self,x,frequency):
        for process in self.preprocessing["dataset"][frequency]:
            x = process(x)
        return x
    
    def lookback_build(self):
        if self.lookback_activated:
            raise Exception("Class PhD4_dataset.lookback_build(): lookback has already been built.")
        
        # import pdb; pdb.set_trace()
        ## unsqueeze data into dataframe
        self.unsplitter(rebatch = False)
        self.dfer()
        
        for frequency in self.input_frequencies:
            # frequency = "w"
            ## sort data
            sort_order = [self.identifier,self.date_entity[frequency]]
            self.store_x[frequency]["train"].sort_index(level = sort_order,inplace=True)
            self.store_index[frequency]["train"] =\
                self.store_index[frequency]["train"].sortlevel(level = sort_order)[0]
            ## reorder index levels
            self.store_index[frequency]["train"] = pd.DataFrame(index=self.store_index[frequency]["train"]).\
                reorder_levels(sort_order).index
            if frequency == self.main:
                self.store_y["train"].sort_index(level = sort_order,inplace=True)
        
            ## create lagged series of table
            self.store_x[frequency]["train"] = [self.store_x[frequency]["train"]]
            for index in range(self.lookback[frequency][1],self.lookback[frequency][0],self.lookback[frequency][1]):
                # index = self.lookback[frequency][1]
                self.store_x[frequency]["train"].append(
                    self.store_x[frequency]["train"][0].groupby(level=self.identifier).\
                        shift(index).to_numpy())
            self.store_x[frequency]["train"][0] = self.store_x[frequency]["train"][0].to_numpy()
            self.store_x[frequency]["train"] = np.stack(self.store_x[frequency]["train"],axis=2)
            
            # import pdb; pdb.set_trace()
            keep_indices = (np.isnan(
                self.store_x[frequency]["train"][:,:,self.lookback_min[frequency]-1]).sum(axis=1)==\
                    0)#self.store_x[frequency]["train"].shape[1])
            # keep_indices*= np.arange(len(keep_indices))
            self.store_x[frequency]["train"] = self.store_x[frequency]["train"][keep_indices,:,:]
            self.store_index[frequency]["train"] = self.store_index[frequency]["train"][keep_indices]
            if frequency == self.main:
                
                self.store_y["train"].reset_index(inplace=True)
                self.store_y["train"] = self.store_y["train"][keep_indices].set_index(sort_order)
                self.store_y["train_backup"] = self.store_y["train"].copy()
                self.store_y["train"] = self.store_y["train"][[
                    self.target_variable,*self.side,*self.meta_list]].to_numpy()
        
        pointer_frequencies = self.input_frequencies+[]
        pointer_frequencies.remove(self.main)
        keep_indices = np.arange(self.store_y["train_backup"].shape[0])
        reference_indices = {frequency:None for frequency in pointer_frequencies}
        drop_indices = []
        for frequency in pointer_frequencies:
            # frequency = "m"
            pointer_column = f"{frequency:}_date"
            
            ## load the pointer table to join with target frequency
            reference_indices[frequency] = self.store_y["train_backup"][[pointer_column]].reset_index()
            reference_indices[frequency].set_index([self.identifier,pointer_column],inplace=True)
            
            ## load the pointed table of the frequency
            index_pointer = pd.DataFrame(data =0,
                index=self.store_index[frequency]["train"],columns = ["index"])
            index_pointer.index.names = [self.identifier,pointer_column]
            index_pointer["index"] = np.arange(index_pointer.shape[0])
            reference_indices[frequency] = pd.merge(
                reference_indices[frequency],index_pointer,
                # left_on = [self.identifier,pointer_column], right_on = [self.identifier,"date"],
                left_index=True, right_index=True,
                how="left")
            reference_indices[frequency].reset_index(inplace=True)
            drop_indices.extend(reference_indices[frequency][reference_indices[frequency]["index"].isna()].index)
            ## some observations have to be dropped because there are not enough lookback observations to back it up.
        keep_indices = list(set(keep_indices).difference(set(drop_indices)))
        
        ## drop observations unmatched because of insufficent lookback from self.main
        self.store_x[self.main]["train"] = self.store_x[self.main]["train"]\
            [keep_indices,...]
        self.store_y["train"] = self.store_y["train"][keep_indices,...]
        self.store_index[self.main]["train"] = \
            self.store_index[self.main]["train"][keep_indices]
        # import pdb; pdb.set_trace()
        self.store_y["train_backup"] = self.store_y["train_backup"].loc\
            [self.store_index[self.main]["train"],:]
        
        ## drop_indices of non-main frequencies
        for frequency in pointer_frequencies:
            # frequency = "m"
            indices = reference_indices[frequency].loc[keep_indices,"index"].astype(int).tolist()
            self.store_x[frequency]["train"] = self.store_x[frequency]["train"][indices,...]
            self.store_index[frequency]["train"] = self.store_index[self.main]["train"].copy()#[indices]
        self.dfed = False
        self.lookback_activated = True
    
    def get_props(self,frequency,subset):
        while self.locked>0:
            time.sleep(.5)
        return self.props[frequency][subset]
    
    def get_data(self,frequency,split,batch=-1,device=device):
        while self.locked>0:
            time.sleep(.5)
        
        batch_calc = batch
        if batch == -1:
            batch_calc = self.props[frequency][split]["batch"]
        # index0 = batch_calc*self.batch_size
        # index1 = min(self.props[frequency]["train"]["len"],index0+self.batch_size)
        # x_train = self.store_x[frequency]["train"][index0:index1,...]
        batch_req = batch_calc%self.props[frequency][split]["n_batches"]
        
        if self.merged:
            x_train = {frequency:self.store_x[frequency][split][device][batch_req]}
        else:
            x_train = {frequency_:self.store_x[frequency_][split][device][batch_req]
                       for frequency_ in self.store_x.keys()}
        y_train = None
        meta = {}
        # import pdb; pdb.set_trace()
        if frequency == self.main:
            # y_train = self.store_y["train"].loc[index0:index1-1,:].\
            #     set_index(self.y_index_columns,drop=True)[[self.target_variable,*side]]
            # y_train = y_train.to_numpy()
            y_raw = self.store_y[split][device][batch_calc%self.props[frequency][split]["n_batches"]]
            if self.meta_series!={}:
                for frequency_ in self.meta_series.keys():
                    index = len(self.side)+self.meta_list.index(self.meta_series[frequency_])+1
                    meta[frequency_] = y_raw[
                        :,index:index+1]
            y_train = y_raw[:,0]
            del y_raw
        if batch == -1:
            self.props[frequency][split]["batch"] +=1
            if self.props[frequency][split]["batch"] == self.props[frequency][split]["n_batches"]:
                self.props[frequency][split]["batch"] = 0
                if self.verbose == "test":
                    print(self.time)
        return x_train, y_train, meta
    
    def get_train(self,frequency,batch=-1, device=device):
        return self.get_data(frequency,"train",batch,device)
    
    def get_val(self,frequency,batch=-1, device=device):
        return self.get_data(frequency,"val",batch,device)
    
    def get_test(self,frequency,batch=-1, device=device):
        return self.get_data(frequency,"test",batch,device)
    
    def get_appl(self,frequency,batch=-1, device=device):
        return self.get_data(frequency,"appl",batch,device)
                
    def get_test_y(self,frequency,side=[]):
        # import pdb; pdb.set_trace()
        y = self.store_y["test_df"][[self.target_variable,*side]]
        return y
    
    def get_test_x(self,frequency,batch=-1):
        '''
        Returns batch of test x data
        '''
        batch_calc = batch
        if batch == -1:
            batch_calc = self.props[frequency]["test"]["batch"]
        if True:#self.mode_data == "full":
            index0 = batch_calc*self.batch_size_testval
            index1 = min(self.props[frequency]["test"]["len"],index0+self.batch_size_testval)
            x_test = self.store_x[frequency]["test"][index0:index1,...]
        if batch == -1:
            self.props[frequency]["test"]["batch"] +=1
            if self.props[frequency]["test"]["batch"] == self.props[frequency]["test"]["n_batches"]:
                self.props[frequency]["test"]["batch"] = 0
                if self.verbose == "test":
                    print(self.time)
        return x_test
    
    def reset_batch_count(self,frequency):
        for subset in ["train","val","test","appl"]:
            self.props[frequency][subset]["batch"] = 0
    
    def shuffle(self):
        '''
        Shuffles training and validation data_points.
        '''
        # import pdb; pdb.set_trace()
        self.locked = 1
        self.shuffle_iteration +=1
        if self.mode_validation == "vw":
            pass
        elif self.mode_validation in ["tocv","rcv"]:
            
            ## select dates for shuffles
            train_val_split = {"train":[],"val":[]}
            date_entity = self.date_entity[self.main]
            
            # determine date entity
            
            distinct_dates = []
            date_entity_shape = {}
            for _split in ["train","val"]:
                # _split = "train"
                dates_split = list(self.store_index[self.main][_split].\
                                   get_level_values(date_entity).drop_duplicates())
                date_entity_shape[_split] = len(dates_split)
                distinct_dates.extend(dates_split)
            del dates_split
            if self.mode_validation == "rcv":
                rd.shuffle(distinct_dates)
            elif self.mode_validation == "tocv":
                distinct_dates = [*distinct_dates[date_entity_shape["val"]:],
                                  *distinct_dates[:date_entity_shape["train"]]]
                
            train_val_split = {
                "train": distinct_dates[:date_entity_shape["train"]],
                "val": distinct_dates[-date_entity_shape["val"]:]}
            
            self.unsplitter()
            self.splitter(rebatch = True,train_val_split = train_val_split)
        self.locked = 0
    def load_spec(self,identifier=("2010-05-23","AAPL")):
        '''
        Allows to return specific datapoint for testing.
        '''
        pass
    
    def datapoint_heatmap(self,frequency="m",_split="train",scale = 3):
        # import pdb; pdb.set_trace()
        if _split == "train":
            batch = np.random.randint(0,len(self.store_x[frequency]["train"][devices[0]]))
            batch_index = np.random.randint(0,len(self.store_x[frequency]["train"][devices[0]][batch]))
            datapoint = self.store_x[frequency]["train"][devices[0]][batch][batch_index,...]
            index = self.store_index[frequency]["train"][devices[0]][batch*self.batch_size+batch_index]
        else:
            batch = np.random.randint(0,len(self.store_x[frequency][_split][devices[0]]))
            batch_index=None
            index = self.store_index[frequency][_split][batch]
            datapoint = self.store_x[frequency][_split][batch]
        if type(datapoint)==to.Tensor:
            datapoint = datapoint.cpu().detach().numpy()
        plt.figure(figsize=(6, 6),facecolor="beige")
        plt.imshow(np.squeeze(datapoint),cmap='PiYG',interpolation = "nearest", aspect='auto',
                   vmin =-scale,vmax=scale)
        plt.colorbar()
        plt.show()
        return {"index":index, "batch":batch, "batch_index":batch_index}
