import numpy as np 

# Convert dBm -> Watts   
def dBm(dBm):
    return 10**((dBm-30)/10)

# Convert dB -> real value 
def dB(dB):
    return 10**(dB/10)

# convert real value -> dB
def to_dB(x):
    return 10*np.log10(x) 

# convert MHz -> Hz 
def MHz(Mhz):
    return Mhz*10**6 

# convert GHz -> Hz 
def GHz(GHz):
    return GHz*10**9

# convet msec -> seconds
def msec(msec):
    return msec*10**(-3)

# convert kbits -> bits 
def Mbits(Mbits):
    return Mbits*10**6 

# convert mW -> W 
def mW(mW):
    return mW*10**(-3) 

# Normalize function 
def normalize(x0):
    return x0/np.linalg.norm(x0, ord=1)


'''
--------------------------------------------------------------------------------
Save and load data to/from a file 
https://www.askpython.com/python/examples/save-data-in-python
--------------------------------------------------------------------------------
'''
import pickle           # used for saving/loading data to/from a file 

def save_data(obj, filepath):
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_data(filepath):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

'''
--------------------------------------------------------------------------------
Plotting figures 
--------------------------------------------------------------------------------
'''  
def plot_moving_average( raw_data, rolling_intv, ylabel, filepath, title=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    data_array = np.asarray(raw_data)
    df = pd.DataFrame(raw_data)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))

    plt.plot(np.arange(len(data_array))+1, np.hstack(df.rolling(window=1, min_periods=1).mean().values), 'b', linewidth=0.5, label='Raw Data')
    plt.plot(np.arange(len(data_array))+1, np.hstack(df.rolling(window=rolling_intv, min_periods=1).mean().values), 'r', label='Moving Average (w={x})'.format(x=rolling_intv))
    plt.fill_between(np.arange(len(data_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.legend()
    plt.title(title)
    plt.savefig(filepath + '/' + ylabel + '.png')
    
def export_moving_average(raw_data, rolling_intv=20):
    import matplotlib.pyplot as plt
    import pandas as pd 
    import matplotlib as mpl
    
    df = pd.DataFrame(raw_data)
    y_axis = np.hstack(df.rolling(window=rolling_intv, min_periods=1).mean().values)
    
    return y_axis

