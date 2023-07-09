import numpy as np 
from utils import *                 # Load utility functions 
from system_paras import * 

rng = np.random.default_rng()

class ArrivalTask:
    def __init__(self, Amean_vec, num_users, time_max, slot_len) -> None:
        '''
        Amean_vec: average arrival rate of tasks in one second, shape = (num_users,)
        '''
        self.Amean_vec = Amean_vec              # shape = (num_users,)
        self.num_users = num_users 
        self.time_max = time_max
        self.slot_len = slot_len
        if task_mode == 'gaussian':
            # arrpox. Poisson as Normal dist. since lambda is large (>=100)
            scale = 1e6
            mean_sc = np.expand_dims(Amean_vec, axis=1)*slot_len/scale # in kBits
            stddev_kB = np.sqrt(mean_sc) 
            arrival_task = (mean_sc + stddev_kB * rng.standard_normal((num_users, time_max)))*scale # shape = (num_users, time_max)
            self.arrival_task = np.where(arrival_task>0, arrival_task, 1)
        
        elif task_mode == 'uniform': 
            self.arrival_task = rng.uniform(low=0, high=np.mean(Amean_vec*slot_len)*2, size=(num_users, time_max))
        
        elif task_mode == 'poisson': 
            ld = 5 # average number of envents in an interval (e.g., a time frane) 
            task_arrivals = np.zeros((num_users, time_max))
            arrival_per_event = Amean_vec*slot_len/ld  # assuming that the arrival workload per event is   
            for i in range(num_users): 
                task_arrivals[i,:] = arrival_per_event[i] * rng.poisson(lam=ld, size=(time_max,))    
            self.arrival_task = task_arrivals 
        
        
def gen_tasks():
    filename = "tasks, {task_mode}, Amean={A:.1f}-{B:.1f} Mbps, (time={t1}s, slot={t2:.2}s).pickle".format(t1= total_time, t2=slot_len, A=np.min(Amean_vec/1e6), B=np.max(Amean_vec/1e6), task_mode=task_mode)
    filepath = os.path.join(tasks_folder, filename)
    if os.path.exists(filepath)==True:
        import warnings
        warnings.warn(f'Data of arrival tasks existed, filepath="{filepath}"')
    else:
        tasks = ArrivalTask(Amean_vec=Amean_vec, num_users=num_users, time_max= time_max, slot_len= slot_len)
        save_data(tasks, filepath)
        print(f'Generated tasks successfully, filepath="{filepath}"')


if __name__ == "__main__":
    gen_tasks()