import numpy as np 
import matplotlib.pyplot as plt
import os 

from utils import *                 # Load utility functions 
from system_paras import *          # Load system parameters 

rng = np.random.default_rng()

# Calculate the next position of user i, (r2,varphi2), based on the previous location,
# (r1, varphi1), the movement direction (phi), and the freedom of movement (stddev_delta_varphi)
def update_location(phi, r1, varphi1, stddev_delta_varphi, mean_delta_r):
    stddev_distance = mean_delta_r/3 
    delta_r = mean_delta_r + stddev_distance * rng.standard_normal()
    delta_r = np.where(delta_r > 0, delta_r, 0)
    
    # delta_varphi = mean_delta_varphi *np.pi + stddev_delta_varphi * rng.standard_normal() # normally distributed with mean=0 and variance=(stddev_delta_varphi)**2
    delta_varphi = (phi-varphi1) + stddev_delta_varphi * rng.standard_normal() # normally distributed with mean=(phi-varphi1) and variance=(stddev_delta_varphi)**2
    if delta_varphi > np.pi:        # normalize so that delta_varphi in range [0,2pi]
        delta_varphi -= 2*np.pi 
    elif delta_varphi < -np.pi:
        delta_varphi += 2*np.pi 
    
    r2 = np.sqrt( r1**2 + delta_r**2 + 2*r1*delta_r*np.cos(delta_varphi) ) 
    
    theta = np.arccos(  (r1**2+r2**2-delta_r**2)/(2*r1*r2) )   # angle between vectors r1 and delta_r, could be negative (i.e., a clockwise direction) 
    varphi2 = np.where( delta_varphi>0, varphi1+theta, varphi1-theta )
    if varphi2 > np.pi:             # normalize so that varphi in range [0,2pi]
        varphi2 -= 2*np.pi 
    elif varphi2 < np.pi: 
        varphi2 += 2*np.pi
    
    return (r2,varphi2)


class User:
    def __init__(self, phi, r0, stddev_delta_varphi, varphi0, mean_delta_r): 
        self.arrival_task = np.zeros(time_max)
        self.qlen_thres = Mbits(1000)   # threshold for long-term average of qlen (qlen < qlen_thres), initiated as infinite 
        
        # Movement: location = (radius, varphi) in the cylinderic coordinate
        self.phi = phi                      # movement direction
        self.radius = np.zeros(time_max) 
        self.varphi = np.zeros(time_max)
        self.varphi[0] = varphi0            # initial varphi 
        self.radius[0] = r0                 # initial radius  
        self.mean_delta_r = mean_delta_r 
        for t in range(1, time_max):
            if isUserLocationFixed == False:
                (r2,varphi2) = update_location(phi=self.phi, r1=self.radius[t-1], \
                    varphi1=self.varphi[t-1], stddev_delta_varphi=stddev_delta_varphi, 
                    mean_delta_r=self.mean_delta_r)
                self.radius[t] = r2 
                self.varphi[t] = varphi2
            else: 
                self.radius[t] = self.radius[t-1]
                self.varphi[t] = self.varphi[t-1]
                
            
        # Channel gain of the user-UAV link 
        theta = np.arctan(uav_altitude / self.radius)       # elevation angle btw user and uav 
        PLOS = 1/( 1 + a_LOS*np.exp( - b_LOS * ( theta - a_LOS ) ) )   # size = (time_max,)
        fading = dB( (mean_fading_log + stddev_fading_log * rng.standard_normal(time_max)) )
        self.channel_gain_nofading = ( PLOS + zeta_LOS*(1-PLOS) ) * g0  \
            / ( uav_altitude**2 + self.radius**2 )**(gamma/2) 
        self.channel_gain = self.channel_gain_nofading * fading
        
        
        # Channel gain of the user-mBS link 
        fading_BS = dB( (mean_fading_log_BS + stddev_fading_log_BS * rng.standard_normal(time_max)) )
        self.channel_gain_BS = fading_BS * g0 / d_macroBS**gamma
                    
        self.cpu_frequency = np.zeros(time_max)                 
        self.tasks_computed_locally = np.zeros(time_max)
        self.tasks_offloaded_to_server = np.zeros(time_max)     
        
        self.power_local_computation = np.zeros(time_max)
        self.power_transmit = np.zeros(time_max)
        self.pw_total = np.zeros(time_max)

        self.queue_length = np.zeros(time_max)
        # self.queue_length[0] += 1               # at time = 0, 1 bit in queue 
        self.vq_qlen_penalty = np.zeros(time_max)
        # self.vq_qlen_penalty[0] += 1            # at time = 0, 1 bit in queue 


    ##### Compute tasks locally: 
    def opt_fcpu_local(self, t, tasks_backlog, VQ_local_i):
        '''
        Arguments:
        - tasks_backlog = tasks (in the queue at the end of the previous slot) - tasks (offloaded)
        - VQ_local = virtual queue for penalty if qlen > a specific threshold
        Return: 
        cpu_frequency, tasks_computed_locally, power_local_computation
        '''
        tasks_backlog_sum = tasks_backlog + VQ_local_i
        cpu_freq_optimal = np.min([ np.min([fcpu_max, tasks_backlog * cycles_per_bit / slot_len]), 
                                np.sqrt(tasks_backlog_sum*slot_len/(3*kappa*Vlyapunov*cycles_per_bit)) ])  # select the optimal cpu frequency
        pw_computation_local  = kappa*cpu_freq_optimal**3        # power consumption for local computation 
        tasks_computed_locally = slot_len*cpu_freq_optimal/cycles_per_bit   # update task computed locally
        return (cpu_freq_optimal, pw_computation_local, tasks_computed_locally)
        
    
    ##### Update the queue length after task computation and task offloading 
    def update_queue(self, t): 
        if t+1 < time_max:          # so that (t+1) <= (time_max-1)
            self.queue_length[t+1] = self.arrival_task[t] + np.max([0, self.queue_length[t] - \
                (self.tasks_computed_locally[t] + self.tasks_offloaded_to_server[t])])
            self.vq_qlen_penalty[t+1] = np.max([0, self.vq_qlen_penalty[t] + scale_vq * (self.queue_length[t+1] - self.qlen_thres)])

    ##### Update power consumption, note: power, not energy -> do not count slot_len 
    def update_power(self, t):
        self.pw_total[t] = self.power_local_computation[t] + self.power_transmit[t]  


def gen_users():
    pickle_fn = "users (time={t1}s, slot={t2:.2}s).pickle".format(t1=total_time, t2=slot_len, A=Amean/1e6)
    chgain_fn = "users-channel-gain (time={t1}s, slot={t2:.2}s).png".format(t1=total_time, t2=slot_len)
    locations_fn = "users-location (time={t1}s, slot={t2:.2}s).png".format(t1=total_time, t2=slot_len)
    
    if os.path.exists(os.path.join(users_folder, pickle_fn))==True:
        import warnings
        warnings.warn(f'Data of users existed, filepath = "{os.path.join(users_filepath, pickle_fn)}"')
    else: 
        users = []                  # list of users 
        list_of_users = []          # list of users' properties 
        for i in range(num_users):
            r0 = rng.integers(50,150)       # initial radical distance 
            varphi0 = rng.random()*2*np.pi  # initial angular coordinate 
            phi = rng.random()*2*np.pi      # movement direction 
            mean_delta_r0 = mean_velocity*slot_len
            list_of_users.append( (r0, varphi0, phi, mean_delta_r0) )

        # generate users
        for idx, (r0, varphi0, phi, mean_delta_r) in enumerate(list_of_users):
            users.append( User(phi=phi, r0=r0, 
                               stddev_delta_varphi=stddev_delta_varphi, 
                                varphi0=varphi0, 
                                mean_delta_r=mean_delta_r
                                ) )
            
        # Save data to a pickle file 
        filepath = os.path.join(users_folder, pickle_fn)
        save_data(users, filepath)
        print('Generated users successfully, filepath="{}"'.format(filepath))
        
        # For plotting figures 
        lines_color = ['-b', '-g', '-r', '-c', '-m', '-k']
        lines_color_nofading = ['.b', '.g', '.r', '.c', '.m', '.k']
        dots_color = ['ob', 'og', 'or', 'oc', 'om', 'ok']
        n_plot = 3       # must be strictly less than len(lines_color)
        
                
        ##### Test 1 : plot radical distance ri(t) versus time 
        # fig1 = plt.figure()
        # for idx, user in enumerate(users):
        #     plt.plot(range(0,time_max),user.radius,lines_color[idx], label=f"users[{idx}]") 
        # plt.xlabel('Time')
        # plt.ylabel('Radical distance, r(t)')
        # plt.grid(True)
        # # fig1.show()
        # plt.savefig('radical distance.png')


        ##### Test 2 : plot real-time locations of users on the ground 
        plt.figure()    # create a figure 
        for idx, user in enumerate(users):
            if idx >= n_plot:
                break
            x = user.radius*np.cos(user.varphi)
            y = user.radius*np.sin(user.varphi)
            plt.plot(x, y, lines_color[idx], label=f"users[{idx}]")
            plt.plot(x[0], y[0], dots_color[idx])
        plt.plot(0, 0, 'ok',label=f"UAV")
        plt.legend()
        plt.grid(True)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig(os.path.join(users_folder, locations_fn))


        ##### Test 3 : Plotting radical distance, r_i(t), and locations of users, (x_i,y_i), wrt time  
        # fig, (ax1,ax2) = plt.subplots(1,2)
        # for idx, user in enumerate(users):
        #     ax1.plot(range(0,time_max),user.radius,lines_color[idx], label=f"users[{idx}]")
        # ax1.grid(True)
        # ax1.set_xlabel('Time')
        # ax1.set_ylabel('Radical distance, r(t)')
        # for idx, user in enumerate(users):
        #     x = user.radius*np.cos(user.varphi)
        #     y = user.radius*np.sin(user.varphi)
        #     # plt.plot(range(0,time_max),users[0].radius,'-r') 
        #     ax2.plot(x, y, lines_color[idx], label=f"users[{idx}]")
        #     ax2.plot(x[0], y[0], dots_color[idx])
        # ax2.grid(True)
        # ax2.set_xlabel('x (m)')
        # ax2.set_ylabel('y (m)')
        # fig.show()
        # fig.savefig(dir_name + f'/location_merge.png')


        ##### Test 4 : plotting channel gain, h_i(t) 
        plt.figure()
        tmax = -1
        for idx, user in enumerate(users):
            if idx >= n_plot:
                break
            plt.plot(range(0,time_max)[:tmax], to_dB(user.channel_gain[:tmax]), lines_color[idx], label=f"users[{idx}], h-UAV", linewidth=0.5) 
            # plt.plot(range(0,time_max)[:tmax], to_dB(user.channel_gain_nofading[:tmax]), lines_color_nofading[idx], markersize=2) 
            plt.plot(range(0,time_max)[:tmax], to_dB(user.channel_gain_BS[:tmax]), lines_color[idx], markersize=2) 
        plt.xlabel("Time")
        plt.ylabel('Channel gain, h(t)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(users_folder, chgain_fn))


if __name__ == "__main__":
    gen_users()