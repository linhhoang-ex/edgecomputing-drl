from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from system_paras import *

print('tensorflow version:', tf.__version__)
print('keras version:', tf.keras.__version__)


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        kernal_size = 4,
        learning_rate = 0.01,
        training_interval=10, 
        batch_size=100, 
        memory_size=1000,
        loss_compute_interval=10,
        epochs = 1,
        output_graph=False
    ):

        self.net = net                              # the size of the DNN, e.g., net = [num_users*kernal_size, 256, 128, num_users*2]
        self.kernal_size = kernal_size,
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.loss_compute_interval = loss_compute_interval
        self.epochs = epochs
        
        # store all binary actions
        self.enumerate_actions = []

        # store # memory entry
        self.memory_counter = 0

        # store training cost
        self.cost_his = []
        
        # test loss
        self.test_loss_memory = np.zeros(loss_compute_interval)
        self.test_cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))  # save the input and output of the DNN in the memory

        # construct memory network
        self._build_net()
        

    def _build_net(self):
        kz = int(self.kernal_size[0])
        self.model = keras.Sequential([
                    layers.Conv1D(32, kz, activation='relu', input_shape=(int(self.net[0]/kz), kz)), # kz = kernal size
                    layers.Conv1D(64, 2, activation='relu'),        
                    layers.Conv1D(128, 2, activation='relu'),       
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),  
                    layers.Dense(64, activation='relu'),
                    layers.Dense(self.net[-1], activation='sigmoid')
                ])
        # self.model = keras.Sequential([
        #             layers.Conv1D(32, kz, activation='relu', input_shape=(int(self.net[0]/kz), kz)), # first Conv1D with 32 channels and kernal size kz
        #             layers.Conv1D(64, 2, activation='relu'), # second Conv1D with 64 channels and kearnal size kz
        #             layers.Conv1D(64, 2, activation='relu'), # second Conv1D with 64 channels and kearnal size 2
        #             layers.Flatten(),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(self.net[-1], activation='sigmoid')
        #             # layers.Dense(self.net[1], activation='relu'),  # the first hidden layer
        #             # layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
        #             # layers.Dense(self.net[-1], activation='sigmoid')  # the output layer
        #         ])

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), 
                           loss=tf.losses.binary_crossentropy, 
                           metrics=['accuracy']
                           )


    def encode(self, h, m, m_pred):
        """
        h : input of the DNN, shape = (self.net[0],)
        m : the optimal action, shape = (self.net[-1],)
        m_pred : output of the DNN, shape = (self.net[-1],)
        """
        # remember the test loss 
        idx0 = self.memory_counter % self.loss_compute_interval
        bce = tf.losses.BinaryCrossentropy()
        self.test_loss_memory[idx0] = bce(m, m_pred).numpy()
        # test_loss_tmp = -(m * np.log(m_pred) + (1 - m) * np.log(1 - m_pred))
        
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1
        
        # train the DNN after every training_interval slots
        # if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter > 0 and self.memory_counter % self.training_interval == 0:
            self.learn()
            if self.memory_counter > self.loss_compute_interval:
                self.test_cost_his.append(np.mean(self.test_loss_memory))
            else:
                self.test_cost_his.append(np.mean(self.test_loss_memory[:self.memory_counter]))
               
        
    def learn(self): 
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        h_train = batch_memory[:, 0: self.net[0]]               # self.net[0] refers to the size of the first layer
        if isSklearnScalerActivated==True:                     # (1) normalize or (2) standardize the input
            # scaler.fit(h_train)     
            scaler.fit(self.memory[:, 0: self.net[0]])                                
            h_train = scaler.transform(h_train)                 
        kernal_size = int(self.kernal_size[0])
        h_train = h_train.reshape(-1, int(self.net[0]/kernal_size), kernal_size)
        m_train = batch_memory[:, self.net[0]:]
    
        # train the DNN
        hist = self.model.fit(x = h_train, 
                              y = m_train,
                            #   batch_size = self.batch_size,
                            #   shuffle = True,
                            #   epochs = self.epochs, 
                              verbose=0
                              )
        self.cost = hist.history['loss'][0]
        assert(self.cost > 0)
        self.cost_his.append(self.cost)


    def decode(self, h, k = 1, num_users = 10): 
        '''
        h : output of the CNN 
        k : # of sample actions to generate
        '''
        
        # to have batch dimension when feed into tf placeholder
        h = h.reshape(num_users,-1)
        h = h[np.newaxis, :]                # equivalent to h = h.reshape(1, num_users, -1) 

        m_pred = self.model.predict(h)      # h.shape = (1, num_users, kernal_size)
        assert np.all(m_pred >= 0) and np.all(m_pred <= 1), "m_pred must be in range (0,1)"

        return (m_pred, self.gen_actions(m_pred[0], k))   # generate a list of k binary feasible actions
    
    
    def knm(self, m, k = 1):
        '''
        return k order-preserving binary actions
        '''
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision 
        m_list.append(1*(m>0.5))
        
        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions 
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list
    
    
    def gen_one_action(self, m):
        '''
        m : the probability vector (output of the CNN)
        return one feasible offloading decision 
        '''
        n_users = num_users
        limit_uav = limit_channel_UAV
        limit_bs = limit_channel_BS
 
        alpha = m[:n_users].copy()
        idx_list = alpha.argsort()[::-1][:limit_uav]
        alpha[:] = 0; alpha[idx_list] = 1
            
        beta = m[n_users:].copy()
        if dual_connectivity == False:     # if dual connectivity is not used, one user can connect to only one MEC server at a time 
            beta[alpha==1] = -np.inf 
        idx_list = beta.argsort()[::-1][:limit_bs]
        beta[:] = 0; beta[idx_list] = 1
        if dual_connectivity == False: 
            assert np.sum(alpha*beta) == 0, 'Dual connectivity not used: alpha and beta cannot be 1 simultaneously'
        return np.hstack((alpha, beta)) 
        
        
    def gen_actions(self, m, k):
        m_list = []
        m_list.append(self.gen_one_action(m))       # genererate the first action, purely based on the output of the NN  
        if k > 1 :    # generate the remaining K-1 actions, based on balance between exploration and exploitation
            for i in range (k-1):
                m_list.append(self.gen_one_action( m + np.random.normal(loc=0, scale=stdvar_gen_action, size=(len(m),)) ))
        return np.unique(m_list, axis=0) 
    
    
    def opn(self, m, k= 1):
        return self.knm(m,k) + self.knm(m+np.random.normal(0,1,len(m)),k)
    
    
    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]
        

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()    # create a figure 
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
