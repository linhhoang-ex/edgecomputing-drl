import numpy as np
from numpy.random.mtrand import beta 
from scipy.optimize import minimize, differential_evolution, shgo
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from utils import *                     # Load utility functions  
from system_paras import *              # Load system parameters
from user import *                      # definintion of class User  


class Server: 
    def __init__(self, ncores = 1, nchannels = 1):   
        self.ncores = ncores
        self.nchannels = nchannels
        self.uplink_alloc   = np.zeros((num_users, time_max))       # uplink bandwidth allocation  
        self.downlink_alloc = np.zeros((num_users, time_max))       # downlink bandwidth allcoation 
        self.qlen           = np.zeros((num_users, time_max))       # queue length, for each user 
        self.fcpu = np.zeros((num_users, time_max))                 # cpu cycles allocated for computing each user's offloaded tasks 
        self.pw_comput = np.zeros(time_max)                         # energy consumed for offloaded task computation 
        self.pw_commun = np.zeros(time_max)                         # energy consumed for communication  
        self.vq_qlen_penalty = np.zeros((num_users, time_max))      # virtual queue for penalty if qlen > a specific threshold   
        
    
    def opt_task_offloading(self, Qi_vec, Li_vec, VQ_local, VQ_remote, hi_vec, hi_BS_vec, alpha, beta):
        '''
        Arguments: 
        Qi_vec      : shape = (num_users,), qlen for each user 
        Li_vec      : shape = (num_users,), remote qlen for each user
        hi_vec      : shape = (num_users,), channel gain for each user (UAV link)
        hi_BS_vec   : shape = (num_users,), channel gain for each user (BS link)
        alpha       : shape = (num_users,), uplink association for the UAV link, if =1 -> the user is associated to the UAV link
        beta        : shape = (num_users,), uplink association for the BS link, if =1 -> the user is associated to the mBS link

        Retuns: 
        offvol_opt  : shape=(num_users,), optimal offloading volume for each user
        pTx_user    : shape=(num_users,), power consumed by the user for transmitting the offloaded task
        bw_uav_opt  : shape = (num_users,), uplink bandwidth allocation for each user (UAV link)
        bw_mbs_opt  : shape = (num_users,), uplink bandwidth allocation for each user (mBS link)
        '''        
        V = Vlyapunov 
        tau = slot_len
        psi = psi_user
        
        def find_offvol(bw_tmp, link_association, Qi, Qi_real, Li, W, h, N0):
            '''
            Function: find the optimal offloading volume, given a feasible solution for bw_uav and bw_mbs
            Args: 
                bw_tmp              : a feasible solution for bw_alloc (tmp: temporary)
                link_association    : uplink association for the given link (fixed) 
                Qi                  : qlen for each user, including the virtual qlen 
                Qi_real             : the real qlen for each user, excluding the virtual qlen
                W                   : total bandwidth of the given link (fixed)
                h                   : channel gain for the given link (fixed)
                N0                  : noise power for the given link (fixed)
            Return:
                offvol_opt          : optimal offloading volume for the given link, given bw_tmp
            '''
            offvol_opt = np.zeros(num_users)                    # offvol_opt: optimal setting for the offloading volume  
            for id in range(0,num_users):
                if link_association[id] == 1:
                    if Qi[id] > Li[id]:   # otherwise, offvol_opt[t] = 0  as default     
                        offvol_opt[id] = W * bw_tmp[id] * tau * \
                            np.log2( (Qi[id]-Li[id])*W*bw_tmp[id]*tau*h[id] / (V*psi[id]*np.log(2)*N0) )
                        assert offvol_opt[id]!=np.nan or offvol_opt[id]!=np.inf, "offvol_opt[id] is not a number"
                else:   # Qi[id] <= Li[id]
                    offvol_opt[id] = 0 
            
            # For adjustment if needed 
            capacity = link_association*W*bw_tmp*tau*np.log2(1 + (pTx_max*h)/(N0))     # channel capacity in bits 
            offvol_max = np.min([capacity, Qi_real], axis=0)      # equivalent: offvol_max = np.where(capacity <= Qi_vec, capacity, Qi_vec)   
            offvol_min = link_association*W*BW_ALLOC_MIN*tau*np.log2(1 + (pTx_max*h)/(N0)) 
            
            # Readjust offvol_opt if needed 
            offvol_opt = np.where(offvol_opt >= offvol_min, offvol_opt, offvol_min)  
            offvol_opt = np.where(offvol_opt <= offvol_max, offvol_opt, offvol_max)
            
            return offvol_opt
        
        def find_bw_alloc(offvol_tmp, link_association, W, h, N0):
            '''
            Function: find the optimal solution for bw allocation, given a feasible solution for offvol 
            Args: 
                offvol_tmp      : a feasible solution for offvol (tmp: temporary), note: if link_association = 0, then offvol_tmp should be 0
                link_association: uplink association for the UAV link or the BS link (fixed) 
                h               : channel gain for the given link 
                W               : total bandwidth of the given link
                N0              : total noise of the given link
            Return: 
                feasible        : whether the problem with given offvol_tmp is feasible for solving bw_opt 
                bw_opt          : optimal solution for bw allocation for the given link (UAV/mBS) 
                lambd_opt       : optimal lambda (Langrangian multiplier) for the given link (UAV/mBS) 
            '''
            num_user = np.sum(link_association)   # no. of users that are associated to the link 
            bw_min = BW_ALLOC_MIN                 # minimum for bandwidth allocation
            bw_max = 1 - (num_user-1)*BW_ALLOC_MIN    # maximum for bandwidth allocation  
            
            # In the case that opt UAV link -> opt mBS link, offvol_tmp of the mBS link could be 0 
            # In that case, if offvol_tmp[i] = 0 -> set link_association[i] = 0 
            link_association_cp = link_association.copy()
            link_association_cp[offvol_tmp==0] = 0 
            
            # Find lambd_max and lambd_min 
            right = psi*N0*offvol_tmp*np.log(2)*2**(offvol_tmp/(W*bw_max*tau))/(h*W*tau*bw_max**2)
            left = psi*N0*offvol_tmp*np.log(2)*2**(offvol_tmp/(W*bw_min*tau))/(h*W*tau*bw_min**2)
            lambd_min = np.max(right[link_association_cp==1])
            lambd_max = np.min(left[link_association_cp==1])
            assert lambd_min>0 and lambd_max>0, "Lambda min and max should be positive"
            
            def solve_bw_alloc_given_lambd(lambd, offvol, h, N0):
                '''
                Function: find bw_opt given lambd (Langrangian multiplier)
                Args: 
                    lambd   : lambda (Langrangian multiplier) 
                    offvol  : offloading volume
                    W       : total bandwidth of the given link
                    h       : channel gain for the given link
                    N0      : total noise of the given link   
                Return: 
                    bw_opt: optimal solution for bw allocation for the given link (UAV/mBS), 
                            by solving derivative of the Lagrangian function = 0
                '''
                from scipy.special import lambertw
                bw_thres = offvol/(W*tau*np.log2(1 + (pTx_max*h)/(N0)))    # shape=(num_users,), bw_opt should be larger than bw_thres
                bw_opt = np.zeros(num_users)
                for i in range(num_users):
                    if offvol[i] > 0: 
                        a = offvol[i]*np.log(2)/(W*tau)
                        c = lambd*h[i]*W*tau/(psi[i]*N0*offvol[i]*np.log(2))
                        assert c > 0, "c should be positive"
                        bw_opt[i] = np.max([a/(2*lambertw(a*np.sqrt(c)/2).real), bw_min])
                bw_opt = np.where(bw_opt >= bw_thres, bw_opt, bw_thres)
                
                return bw_opt
            
                        
            # Check wheher the Lagrangian equation is solvable with lambd_max 
            # (i.e., can find a feasible solution for bw_uav so that sum(bw_alloc) <= 1))) 
            # If not sovable with lambd_max -> terminate the WHILE loop 
            feasible = True 
            bw_test = solve_bw_alloc_given_lambd(lambd_max, offvol_tmp, h, N0)
            if lambd_min > lambd_max or np.sum(bw_test) > 1: 
                feasible = False 
            
            # Find the best lambd so that sum(bw_alloc) is closest to 1
            iter_cnt = 0 
            lambd_lb = lambd_min
            lambd_ub = lambd_max 
            lambd_til = 0.5*(lambd_lb+lambd_ub)
            bw_tmp = np.zeros(num_users)
            while(feasible==True and iter_cnt<ITERATION_MAX_lambd):    # loop until the problem is feasible and iter<iter_max 
                lambd_til = 0.5*(lambd_lb+lambd_ub)
                iter_cnt += 1
                bw_tmp = solve_bw_alloc_given_lambd(lambd_til, offvol_tmp, h, N0)
                if np.sum(bw_tmp) > 1: 
                    lambd_lb = lambd_til
                else: 
                    lambd_ub = lambd_til 
                # the constraint sum(bw)<=1 is satisfied -> break the loop  
                if np.abs(np.sum(bw_tmp) - 1) <= epsilon_kkt_lambd: 
                    break 
            
            # Readjust bw_tmp in range [bw_min, bw_max]
            bw_opt = bw_tmp.copy()
            for i in np.where(link_association==1)[0]:
                bw_opt[i] = np.min([bw_tmp[i], bw_max])
                bw_opt[i] = np.max([bw_tmp[i], bw_min])
                
            return (feasible, bw_opt, lambd_til) 
        
        def solve_GS(link_association, Qi, Qi_real, Li, W, h, N0):
            '''
            Use 2 functions find_offvol() and find_bw_alloc() to solve the problem for one link (UAV/mBS)
            Parameters: 
                link_association: uplink association for the UAV link or the BS link (fixed) 
                Qi              : qlen for each user, including the virtual qlen 
                Qi_real         : the real qlen for each user, excluding the virtual qlen
                h               : channel gain for the given link 
                W               : total bandwidth of the given link
                N0              : total noise of the given link
            Returns:
                bw_opt          : optimal solution for bw allocation for the given link (UAV/mBS)
                offvol_opt      : optimal solution for offvol for the given link (UAV/mBS)
            '''
            # if Qi_real=0 -> no need to solve the problem, bw_opt=any, offvol_opt=0
            converged = True if np.all(Qi_real[link_association==1]==0) else False
            
            # Initiate optimal solution for offloading volume (b)
            bw_alloc_0 = link_association/link_association.sum()         # equal bandwidth allocation for UAV link
            offvol_0 = find_offvol(bw_alloc_0, link_association, Qi, Qi_real, Li, W, h, N0)
            
            def cal_fval(Qi, Li, link_association, bw_alloc, offvol, W, h, N0):
                fval = 0 
                for i in range(num_users):
                    if link_association[i]==1:
                        fval += (-1)*(Qi[i]-Li[i])*offvol[i] + V*psi[i]*(2**(offvol[i]/(W*bw_alloc[i]*tau))-1)*(N0/h[i])
                return fval  
            
            # For tracking the best solution so far 
            bw_opt = bw_alloc_0.copy()
            offvol_opt = offvol_0.copy()
            fval_opt = cal_fval(Qi, Li, link_association, bw_opt, offvol_opt, W, h, N0)
            
            iter_cnt = 0      # count the number of iterations
            while converged==False and iter_cnt < ITERATION_MAX_GaussSeidel:
                feasible, bw_alloc_1, lambd = find_bw_alloc(offvol_0, link_association, W, h, N0)
                if feasible==False:
                    break 
                offvol_1 = find_offvol(bw_alloc_1, link_association, Qi, Qi_real, Li, W, h, N0)
                
                # Track the best-so-far solution 
                fval_tmp = cal_fval(Qi, Li, link_association, bw_alloc_1, offvol_1, W, h, N0)
                if fval_tmp < fval_opt:
                    bw_opt = bw_alloc_1.copy(); offvol_opt = offvol_1.copy(); fval_opt = fval_tmp
                
                # Terminate if the solution converges
                converged = np.allclose(bw_alloc_0, bw_alloc_1, atol=atol_bw, rtol=rtol_bw)
                bw_alloc_0 = bw_alloc_1.copy(); offvol_0 = offvol_1.copy()
                iter_cnt += 1
            
            return (fval_opt, bw_opt, offvol_opt)

        # Solve the problem for all links, using the function solve_GS()
        Qi_sum = Qi_vec + VQ_local
        Li_sum = Li_vec + VQ_remote
        fval_uav, bw_uav_opt, offvol_uav_opt = solve_GS(alpha, Qi_sum, Qi_vec, Li_sum, bw_total_uav, hi_vec, sigma_sq_uav)
        fval_mbs, bw_mbs_opt, offvol_mbs_opt = solve_GS(beta, Qi_sum-offvol_uav_opt, Qi_vec-offvol_uav_opt, np.zeros(num_users), bw_total_mbs, hi_BS_vec, sigma_sq_mbs)
        if dual_connectivity == False: 
            assert np.sum(bw_uav_opt*bw_mbs_opt)==0, "Error: one user connect to both UAV and mBS"
            assert np.sum(offvol_uav_opt*offvol_mbs_opt)==0, "DC not activated: one user cannot connect to both UAV and mBS at the same time"
        assert np.sum(bw_uav_opt) <= 1+BW_ALLOC_MIN*num_users, "Error: sum(bw_uav) > 1"
        assert np.sum(bw_mbs_opt) <= 1+BW_ALLOC_MIN*num_users, "Error: sum(bw_mbs) > 1"
        offvol_opt = offvol_uav_opt + offvol_mbs_opt
        
        # calculate pTx for each user 
        pTx_user_uav = np.zeros(num_users)
        pTx_user_mbs = np.zeros(num_users)
        for i in range(0,num_users): 
            if alpha[i] == 1:
                pTx_user_uav[i] = (2**(offvol_uav_opt[i]/(bw_total_uav * bw_uav_opt[i] * tau)) - 1) * (sigma_sq_uav) / hi_vec[i] 
            if beta[i] == 1:
                pTx_user_mbs[i] = (2**(offvol_mbs_opt[i]/(bw_total_mbs * bw_mbs_opt[i] * tau)) - 1) * (sigma_sq_mbs) / hi_BS_vec[i] 
        pTx_user = pTx_user_uav + pTx_user_mbs 
        assert np.all(pTx_user_uav <= pTx_max + 1), "pTx_user_uav > pTx_max"
        assert np.all(pTx_user_mbs <= pTx_max + 1), "pTx_user_mbs > pTx_max"
        
        fval_opt = fval_uav + fval_mbs 
        return (fval_opt, offvol_uav_opt, offvol_mbs_opt, bw_uav_opt, bw_mbs_opt, pTx_user)        # shape = (num_users,)
        
    
    ##### Joint optimization for Remote Computation and Downlink Bandwidth Allocation
    def opt_fcpu_uav(self, Li_vec, VQ_remote):
        '''
        Li_vec: shape = (num_users,), remote qlen for each user
        VQ_remote: shape = (num_users,), remote virtual qlen for penalty if qlen > qlen_thres
        '''
        Li_sum = Li_vec + VQ_remote
        # cpu_freq_optimal = np.min([ np.min([fcpu_core_uav_max, Li_vec * cycles_per_bit / slot_len]), 
        #                            np.sqrt(Li_vec/(3*kappa*Vlyapunov*cycles_per_bit)) ])  # select the optimal cpu frequency
        cpu_freq_optimal = np.sqrt(Li_sum*slot_len/(3*kappa*Vlyapunov*psi_uav*cycles_per_bit))
        cpu_freq_ref = Li_vec * cycles_per_bit / slot_len
        cpu_freq_ub = np.where(cpu_freq_ref <= fcpu_core_uav_max, cpu_freq_ref, fcpu_core_uav_max)
        cpu_freq_optimal = np.where(cpu_freq_optimal < cpu_freq_ub, cpu_freq_optimal, cpu_freq_ub)

        return cpu_freq_optimal
    
    def assign_subchannel_greedy_qlen(self, Qi_vec, Li_vec, VQ_local, VQ_remote):
        '''
        Args: current qlen of users 
        Return: subchannel assignment for each user, alpha: to uav, beta: to mBS 
        '''
        # qlen_sum =  VQ_local + Qi_vec
        qlen_sum =  VQ_local 
        idx_list = qlen_sum.argsort()[::-1][:(limit_channel_UAV+limit_channel_BS)] # sorting in descending order
        alpha = np.zeros(num_users); alpha[idx_list[:limit_channel_UAV]] = 1
        beta = np.zeros(num_users); beta[idx_list[limit_channel_UAV:]] = 1
        return np.hstack((alpha, beta)).reshape(1,-1)
    
    
    def assign_subchannel_greedy_chgain(self, chgain_uav, chgain_mbs):
        '''
        Args: current chgain of users, to uav and to mBS 
        Return: subchannel assignment for each user, alpha: to uav, beta: to mBS         
        '''
        idx_list_uav = chgain_uav.argsort()[::-1][:limit_channel_UAV] # indexes in descending order
        alpha = np.zeros(num_users); alpha[idx_list_uav[:limit_channel_UAV]] = 1
        
        chgain_mbs_copy = chgain_mbs.copy()
        if dual_connectivity == False: 
            chgain_mbs_copy[idx_list_uav[:limit_channel_UAV]] = 0       # users assigned to uav cannot connect to mbs at the same time 
        idx_list_mbs = chgain_mbs_copy.argsort()[::-1][:limit_channel_BS] # indexes in descending order
        beta = np.zeros(num_users); beta[idx_list_mbs[:limit_channel_BS]] = 1
        return np.hstack((alpha, beta)).reshape(1,-1)
    
    
    def gen_actions_bf(self):
        '''
        generate all feasible actions using Brute Force methods 
        ''' 
        import itertools
        brute_force = np.array(list(itertools.product([0, 1], repeat=num_users*2)))
        # print("brute_force.shape = ", brute_force.shape)
        n_col = brute_force.shape[1]
        n1 = limit_channel_UAV; n2 = limit_channel_BS
        x = brute_force[:,:n_col//2]  # shape = (64,3)
        y = brute_force[:,n_col//2:]  # shape = (64,3)
        filter1 = np.sum(x, axis=1) == n1   # shape = (64,)
        filter2 = np.sum(y, axis=1) == n2   # shape = (64,)
        filter3 = np.all(x + y <= 1, axis=1)     # shape = (64,)
        if dual_connectivity == True:
            filter3 = True    # deactivate filter 3 
        bf_solutions = brute_force[filter1 & filter2 & filter3] 
        # print("bf_solutions.shape =", bf_solutions.shape)
        return bf_solutions
    