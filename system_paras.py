import numpy as np 
rng = np.random.default_rng()               # to be used in generating random numbers 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import *                         # self-defined utility functions 


'''
--------------------------------------------------------------------------------
                            System parameters 
--------------------------------------------------------------------------------
'''
num_slots = 15e3                                # number of time slots 
slot_len = msec(10)                             # time slot length (to) in msec. NOTE: should be at least 10ms so that approximation for arrival task is appropriate 
total_time = int(num_slots*slot_len)            # total time in seconds 
time_max = 1 + np.int64(total_time/slot_len)    # number of time slots for indexing
assert num_slots == time_max-1, "num_slots != time_max-1"           

num_users = 8                                   # number of mobile users    
# distance_user_to_server = 150                 # distance from each user to the server in meter (m)  

bw_uplink = MHz(0.2)                            # uplink bandwidth
# bw_downlink = MHz(1)                            # downlink bandwidth 
limit_channel_UAV = 2                           # maximum # of users that can offload tasks to the UAV in one time slot  
limit_channel_BS = 2                            # maximum # of users that can offload tasks to the macro BS in one time slot 
bw_total_uav = bw_uplink * limit_channel_UAV
bw_total_mbs = bw_uplink * limit_channel_BS
BW_ALLOC_MIN = 0.05                             # minimum bandwidth allocation for each user

noise_density = dBm(-174)                       # noise power spectral density (N_0) in dBm/Hz 
sigma_sq_uav = bw_total_uav*noise_density       # noise power of the whole bandwidth in Watt, sigma_sq = sigma^2 
sigma_sq_mbs = bw_total_mbs*noise_density       # noise power of the whole bandwidth in Watt, sigma_sq = sigma^2 
g0 = dB(-50)                                    # path-loss consistant in dBm 
# hi_smallscale_min = 0.1                       # small-scale channel gain h_i(t)   
# distance_ref = 1                              # reference distance between mobile user and MEC server (d_0) in meter (m) 
gamma = 2.7601                                  # path-loss exponent       
mean_fading_log = 0; stddev_fading_log = 4      # for fading of UAV-user communication channel
mean_fading_log_BS = 0
stddev_fading_log_BS = 8 

cycles_per_bit = 737.5                      # number of cpu cycles to compute a bit (cycles/bit)
kappa = 0.1e-27                             # power constant for function power = f(frequency)
warning_enable = False                      # print warnings onto the terminal 


'''
--------------------------------------------------------------------------------
                            Parameters for the user 
--------------------------------------------------------------------------------
'''      
# NOTE: Amean*slot_len should be at least hundreds of kB so that the Possion approxmimation as Gaussian distribution is appropriate 
Amean_vec =  np.ones(num_users)*Mbits(3.0)              # arrival task mean in Mbps
Amean = int(np.mean(Amean_vec))
task_models = ['uniform', 'gaussian', 'poisson']    
#  Code:            0           1          2    
task_mode = task_models[2]   

fcpu_min = 0                                
fcpu_max = GHz(1.5)                         # cpu frequency in GHz for user devices
pTx_min, pTx_max = 0, mW(100)               # transmit power in mW  
b_min = 0                                   # minimum for b_i(t) (tasks offloaded in bits)
init_radius = 30                            # in meter
mean_velocity = 1.5                         # meters per second  
stddev_delta_varphi = np.pi/6               # standard deviation of varphi
d_macroBS = 500                             # distance to macro BS in meter
isUserLocationFixed = True                  # if True, the location of the user is fixed
# mean_delta_r  = mean_velocity*slot_len    # mean of movement distance in a time slot  
# stddev_distance = mean_delta_r/3          # standard deviation of movement distance
# mean_delta_varphi = np.pi/3 


'''
--------------------------------------------------------------------------------
                            Parameters for the UAV 
--------------------------------------------------------------------------------
'''
ncores = num_users                          # number of computation cores of one MEC server 
uav_altitude = 50                           # in meters 
a_LOS, b_LOS = 9.61, 0.16                   # S-curve parameters for calculating PLOS
zeta_LOS = 0.5                              # attenuation effect of NLOS 
nu = 1                                      # size(output)/size(input) for task computation at the UAV 
# pTx_uav_max = mW(500)                       # maximum pTx for uav (for downlink)
fcpu_uav_min = 0                            # CPU frequency (cycles/sec) for the UAV  
fcpu_uav_max = GHz(10)
fcpu_core_uav_max = GHz(1.5)                # maximum frequency for each core of the UAV


'''
--------------------------------------------------------------------------------
                        Parameters for the optimization 
--------------------------------------------------------------------------------
'''
Vlyapunov = 1e9                             # control parameter (V) in the Lyapunov's drift-plus-penalty function 
k_max = 10                                  # maximum of repetitions in Gauss-Seidel method 
converge_cnt_thres = 1                      # same results for converge_cnt_thres -> converged 
eps_gs = 1e-4                               # Gauss-Seidel's terminate tolerence 
atol, rtol = 1e-05, 1e-05                   # absolute and relative tolerance for KKT equations and Gauss-Seidel, absolute(a - b) <= (atol + rtol * absolute(b)) 
exp_thres = 50                              # threshold for exp in 2**exp, 2**50 -> 10**15  
my_inf = dB(200)                            # if exp > exp_thres -> fval = my_inf 

psi_uav = 1/(num_users*3)                   # psi indicates the importance of UAV's energy compared to users' energy
psi_user = np.ones(num_users)               # energy preference when optimizing each user's energy

scale_vq = 1                                # scale for VQ, Z(t+1) = Z(t) + scale_vq * vq(t)     
is_penalty_qlen_used = True                 # if True, the penalty function is used for the queue length if qlen > qlen_thres   


is_qthres_fixed = True                                  # if True, set qlen_thres = qlen_thres_fixed 
is_qthres_infty = True                                  # if True, set qlen_thres = infty           
qlen_thres_user_fixed = 40e3*np.ones(num_users)         # in bits 
qlen_thres_uav_fixed = 5e3*np.ones(num_users)           # in bits 


if is_qthres_fixed == False: 
        if Amean == Mbits(3.0): 
                qlen_thres_scale_user = np.array([1.5]*num_users)*100
                qlen_thres_scale_uav = np.array([0.3]*num_users)*100 
        else: 
                qlen_thres_scale_user = np.array([1.5]*num_users)
                qlen_thres_scale_uav = np.array([0.3]*num_users)
        qlen_thres_user = Amean_vec*slot_len*qlen_thres_scale_user
        qlen_thres_uav = Amean_vec*slot_len*qlen_thres_scale_uav
else:   # qthres_fixed = True 
        qlen_thres_user = qlen_thres_user_fixed
        qlen_thres_uav = qlen_thres_uav_fixed
        
        if is_qthres_infty == True: 
                qlen_thres_user = qlen_thres_user*1000
                qlen_thres_uav = qlen_thres_uav*1000 
        
        

'''
--------------------------------------------------------------------------------
        Parameters for the Gause-Seidel method and KKT Condition
--------------------------------------------------------------------------------
'''        
epsilon_kkt_lambd = 1e-2                    # error tolerence for terminating KKT optimization  
ITERATION_MAX_GaussSeidel = 5               # maximum of iterations for KKT optimization (i.e., finding best lambd)
ITERATION_MAX_lambd = 30 
atol_bw = 1e-3                              # absolute tolerence for Gauss-Seidel method, absolute(a - b) <= (atol + rtol*absolute(b))
rtol_bw = 1e-3                              # relative tolerence for Gauss-Seidel method


'''
--------------------------------------------------------------------------------
                        Parameters for neural network 
--------------------------------------------------------------------------------
'''    
# decoder_mode = 'OPN'                      # the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
CHFACT_UAV = dB(100)                        # The factor for scaling channel value
CHFACT_BS = dB(115)                         # The factor for scaling channel value
QLFACT = 1/(Amean*slot_len)               # The factor for scaling qlen value
VQFACT = 1/(Amean*slot_len)
kernal_size = 6                             # for construction of the convolutional neural network
learning_rate = 1e-3
training_interval = 20 
epochs = 1 
Memory = 1024                               # capacity of memory structure 
batch_size = 256
loss_compute_interval = int(batch_size/4)        
stdvar_gen_action = 0.25                    # standard deviation of action generation
# Delta = 32                                # update interval for adaptive K

# For scaling the input of the neural network
isSklearnScalerActivated = True             # if True, the scaler is activated
scaler = MinMaxScaler()
# scaler = StandardScaler()

# For loading a pretrained model
load_pretrained_model = False               # if True, the pretrained model is loaded 
trained_model_modes = ['learning', 'exhausted search'] 
trained_model_mode = trained_model_modes[0]


# For selecting the sever selection mode and saving the trained model
modes = ['learning', 'exhausted search', 'random', 'greedy (qlen)', 'greedy (chgain)', 'distributed(greedy)']
# Code:        0             1               2           3                  4                 5 
selection_mode = modes[0]          
is_training_model = True if selection_mode == "learning" or selection_mode == "exhausted search" else False   


is_pTxmax_used = False 
load_memory = False 
test_mode = False                        # if True, print details of system statistics in each time slot 
dual_connectivity = True                # if True, one user can simultaneously offload tasks to both the UAV and the BS 
new_task_generation = True              # if True, generate new arrival task and channel gains for each user 

is_n_actions_fixed = False               # if True, n_actions=n_actions_fixed; if False, n_actions = n_actions_scale * search space
n_actions_fixed = 50
n_actions_scale = 0.1 


'''
--------------------------------------------------------------------------------
        For loading data and saving figures/data
--------------------------------------------------------------------------------
'''
import os 

##### For loading data 
users_filepath = os.path.join(os.getcwd(), "in_users", "users (time={t1}s, slot={t2:.2}s).pickle".format(t1=total_time, t2=slot_len))
tasks_filepath = os.path.join(os.getcwd(), "in_tasks", "tasks, {task_mode}, Amean={A:.1f}-{B:.1f} Mbps, (time={t1}s, slot={t2:.2}s).pickle".format(t1= total_time, t2=slot_len, A=np.min(Amean_vec/1e6), B= np.max(Amean_vec/1e6), task_mode=task_mode))
users_folder = os.path.join(os.getcwd(), "in_users")
tasks_folder = os.path.join(os.getcwd(), "in_tasks")

trained_model_filepath =  os.path.join(os.getcwd(), "trained_models", "memoryTF2conv, train={x} by {mode}.json".format(x=total_time, mode=trained_model_mode))
trained_weights_filepath = os.path.join(os.getcwd(), "trained_models", "memoryTF2conv, train={x} by {mode}.h5".format(x=total_time, mode=trained_model_mode))
models_folder = os.path.join(os.getcwd(), "trained_models")

##### For saving data and figure 
path_to_sim_folder = os.path.join(os.getcwd(), "sim")
dir_name = "{opt_mode}, V={V:.1e}, Amean={A} Mbps ({task_mode}), time={time}s, slot={slot}s, lr={lr:.0e}".format(
        opt_mode=selection_mode, V=Vlyapunov, A=Amean/1e6, task_mode=task_mode, time=total_time, slot=slot_len, lr=learning_rate)
mypath = os.path.join(path_to_sim_folder, dir_name)
if os.path.exists(mypath) == False:
    os.mkdir(mypath)

