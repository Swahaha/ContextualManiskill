from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.parallel import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather

from hyperppoCNN_model.ghn_modules import MLP_GHN
from hyperppoCNN_model.model import MlpNetwork

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class hyperActor(nn.Module):

    def __init__(self, 
                act_dim, 
                obs_dim, 
                allowable_layers, 
                meta_batch_size = 1,
                device = "cpu",
                architecture_sampling_mode = "biased",
                multi_gpu = True,
                std_mode = 'multi',
                allow_conv_layers = True,
                max_conv_layers = 2,
                ):
        super().__init__()
        print("Initializing hyperActor...")

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.meta_batch_size = meta_batch_size
        self.architecture_sampling_mode = architecture_sampling_mode
        self.multi_gpu = multi_gpu
        self.allow_conv_layers = allow_conv_layers
        self.max_conv_layers = max_conv_layers
        assert std_mode in ['single', 'multi', 'arch_conditioned'], "std_mode must be one of ['single', 'multi', 'arch_conditioned']"
        self.std_mode = std_mode

        print("Initializing devices...")
        # initialize all devices for parallelization on multiple GPUs
        self._initialize_devices(device)
        print("Devices initialized")

        print("Initializing shape and architecture indices...")
        # initialize all list of shape and architecture indices
        self._initialize_shape_arch_inidices(allowable_layers)
        print("Shape and architecture indices initialized")

        print("Initializing architecture sampling data...")
        # initialize all data required for architecture sampling
        self._initialize_architecture_smapling_data()
        print("Architecture sampling data initialized")

        print("Initializing GHN...")
        # initialize the GHN
        self._initialize_ghn(self.obs_dim, self.act_dim)
        print("GHN initialized")

        print("Initializing standard deviation vectors...")
        # initialize standard deviation vectors
        self._initialize_std()
        print("Standard deviation vectors initialized")
        print("hyperActor initialization complete")

    def _initialize_std(self):
        ''' Initializes the standard deviation vectors
        '''
        if self.std_mode == 'single':
            self.log_std = nn.Parameter(torch.zeros(1, np.prod(self.act_dim)))
        elif self.std_mode == 'multi':
            self.log_std = nn.ParameterList([
                nn.Parameter(torch.zeros(1, np.prod(self.act_dim)), requires_grad = False)
            for index in self.list_of_arc_indices
            ])
            pass
        elif self.std_mode == 'arch_conditioned':
            self.log_std = nn.Sequential(
                    layer_init(nn.Linear(self.arch_max_len, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 1), std=1.0),
                )
        else:
            raise NotImplementedError  


    def _initialize_architecture_smapling_data(self):
        ''' Initializes all the data required for architecture sampling
        '''
        if self.architecture_sampling_mode == "sequential":
            self.current_model_indices = np.arange(self.meta_batch_size)
        elif self.architecture_sampling_mode == "uniform":
            # self.current_model_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace = True)
            pass
        elif self.architecture_sampling_mode == "biased":
            self.arch_sampling_probs = []
            num_unique_num_layers = len(set([len(x) for x in self.list_of_arcs]))
            for i in self.list_of_arc_indices:
                num_layers = len(self.list_of_arcs[i])
                num_archs_with_same_num_layers = len([x for x in self.list_of_arcs if len(x) == num_layers])
                self.arch_sampling_probs.append(1/num_archs_with_same_num_layers)
            self.arch_sampling_probs = (1/num_unique_num_layers)*np.array(self.arch_sampling_probs)

        self.sampled_indices = None





    def _initialize_shape_arch_inidices(self, allowable_layers):
        ''' Creates:
            1. list_of_arcs: list of all possible architectures, sorted by total number of parameters
            2. list of shape indicators: list of shape indicators for each architecture, that can be used as an input to the GHN
            3. list of arc indices: list of indices of the architectures in list_of_arcs, can be used to sample architectures later
        '''
        print("Starting _initialize_shape_arch_inidices...", flush=True)
        list_of_allowable_layers = list(allowable_layers)
        self.list_of_arcs = []
        
        # Define CNN layer options if allowed
        if self.allow_conv_layers:
            print("Adding CNN layer options...", flush=True)
            conv_options = [
                (1, 8, 3),   # First layer: 1 input channel -> 8 output channels
                (8, 16, 3),  # Second layer: 8 input channels -> 16 output channels
            ]
            # Add CNN layers to allowable layers
            list_of_allowable_layers.extend(conv_options)
            print(f"Total allowable layers: {len(list_of_allowable_layers)}", flush=True)

        # Generate architectures
        print("Generating architectures...", flush=True)
        for k in range(1, 5):  # Up to 4 layers
            print(f"Generating architectures with {k} layers...", flush=True)
            for layers in product(list_of_allowable_layers, repeat=k):
                # Validate architecture
                if self.allow_conv_layers:
                    # Count CNN layers
                    conv_count = sum(1 for layer in layers if isinstance(layer, tuple))
                    if conv_count > self.max_conv_layers:
                        continue
                    
                    # Ensure CNN layers are at the start and have correct input channels
                    if conv_count > 0:
                        valid_architecture = True
                        for i, layer in enumerate(layers):
                            if isinstance(layer, tuple):
                                in_channels, out_channels, kernel_size = layer
                                # First CNN layer must have 1 input channel
                                if i == 0 and in_channels != 1:
                                    valid_architecture = False
                                    break
                                # Subsequent CNN layers must match previous layer's output channels
                                if i > 0 and isinstance(layers[i-1], tuple):
                                    prev_out_channels = layers[i-1][1]
                                    if in_channels != prev_out_channels:
                                        valid_architecture = False
                                        break
                                # No MLP layers before CNN layers
                                if i > 0 and not isinstance(layers[i-1], tuple):
                                    valid_architecture = False
                                    break
                        
                        if valid_architecture:
                            self.list_of_arcs.append(layers)
                    else:
                        self.list_of_arcs.append(layers)
                else:
                    self.list_of_arcs.append(layers)
            print(f"Found {len(self.list_of_arcs)} architectures so far...", flush=True)

        print(f"Total architectures generated: {len(self.list_of_arcs)}", flush=True)
        print("Sorting architectures by parameter count...", flush=True)
        self.list_of_arcs.sort(key=lambda x: self.get_params(x))
        print("Architectures sorted", flush=True)

        print("Initializing shape indicators...", flush=True)
        self._initialize_shape_inds()
        print("Shape indicators initialized", flush=True)

        self.list_of_arc_indices = np.arange(len(self.list_of_arcs))
        print("Creating model instances...", flush=True)
        self.all_models = [MlpNetwork(fc_layers=self.list_of_arcs[index], inp_dim=self.obs_dim, out_dim=self.act_dim) 
                          for index in self.list_of_arc_indices]
        print(f"Created {len(self.all_models)} model instances", flush=True)
        
        # shuffle the list of arcs indices
        print("Shuffling architecture indices...", flush=True)
        np.random.shuffle(self.list_of_arc_indices)
        print("_initialize_shape_arch_inidices complete", flush=True)


    def _initialize_shape_inds(self):
        ''' Creates:
            1. list_of_shape_inds: list of shape indicators for each architecture, that can be used as an input to the GHN
            2. list_of_shape_inds_lenths: list of lengths of each shape indicator, needed since the shape indicators are not all the same length
        '''
        print("Starting _initialize_shape_inds...", flush=True)
        self.list_of_shape_inds = []
        print(f"Processing {len(self.list_of_arcs)} architectures...", flush=True)
        
        for i, arc in enumerate(self.list_of_arcs):
            if i % 100 == 0:
                print(f"Processing architecture {i}/{len(self.list_of_arcs)}...", flush=True)
            shape_ind = [torch.tensor(0).type(torch.FloatTensor).to(self.device)]
            
            for layer in arc:
                if isinstance(layer, tuple):  # CNN layer
                    in_channels, out_channels, kernel_size = layer
                    shape_ind.append(torch.tensor(in_channels).type(torch.FloatTensor).to(self.device))
                    shape_ind.append(torch.tensor(out_channels).type(torch.FloatTensor).to(self.device))
                    shape_ind.append(torch.tensor(kernel_size).type(torch.FloatTensor).to(self.device))
                else:  # MLP layer
                    shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
                    shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
            
            shape_ind.append(torch.tensor(self.act_dim).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim).type(torch.FloatTensor).to(self.device))
            shape_ind = torch.stack(shape_ind).view(-1,1)
            self.list_of_shape_inds.append(shape_ind)

        print("Calculating shape indicator lengths...", flush=True)
        self.list_of_shape_inds_lenths = [x.squeeze().numel() for x in self.list_of_shape_inds]
        self.shape_inds_max_len = max(self.list_of_shape_inds_lenths)
        self.arch_max_len = 4
        print(f"Max shape indicator length: {self.shape_inds_max_len}", flush=True)
        
        print("Padding shape indicators...", flush=True)
        # pad -1 to the end of each shape_ind
        for i in range(len(self.list_of_shape_inds)):
            if i % 100 == 0:
                print(f"Padding shape indicator {i}/{len(self.list_of_shape_inds)}...", flush=True)
            num_pad = (self.shape_inds_max_len - self.list_of_shape_inds[i].shape[0])
            self.list_of_shape_inds[i] = torch.cat([self.list_of_shape_inds[i], torch.tensor(-1).to(self.device).repeat(num_pad,1)], 0)
        
        print("Stacking shape indicators...", flush=True)
        self.list_of_shape_inds = torch.stack(self.list_of_shape_inds)
        self.list_of_shape_inds = self.list_of_shape_inds.reshape(len(self.list_of_shape_inds),self.shape_inds_max_len)
        print("_initialize_shape_inds complete", flush=True)


    def _initialize_devices(self, device):
        ''' Inititalize all devices since we are using multiple GPUs. device_model_list can be used later to assign models to devices quickly
        '''
        if self.multi_gpu:
            self.device = torch.device("cuda:0")            
            
            self.all_devices = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
            self.num_current_models_per_device = int(self.meta_batch_size / len(self.all_devices)) 
            self.device_model_list = []
            for device in self.all_devices:
                self.device_model_list.extend([device for i in range(self.num_current_models_per_device)])
        else:
            self.device = device



    def _initialize_ghn(self, obs_dim, act_dim):
        ''' Initialize the GHN that takes in the shape indicators and outputs weights for that corresponding architecture
        '''

        config = {}
        config['max_shape'] = (256, 256, 1, 1)
        config['num_classes'] = 2 * act_dim
        config['num_observations'] = obs_dim
        config['weight_norm'] = True
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = 16
        self.ghn_config = config
        self.ghn = MLP_GHN(**config,
                    debug_level=0, device=self.device).to(self.device)  



    def get_params(self, net):
        ''' Get the number of parameters in a network architecture
        '''
        ct = 0
        current_dim = self.obs_dim
        
        for layer in net:
            if isinstance(layer, tuple):  # CNN layer
                in_channels, out_channels, kernel_size = layer
                # Conv weights: in_channels * out_channels * kernel_size * kernel_size
                ct += (in_channels * out_channels * kernel_size * kernel_size)
                # Conv bias: out_channels
                ct += out_channels
                # Update current dimension for next layer
                current_dim = out_channels * (current_dim // in_channels)
            else:  # MLP layer
                # If previous layer was CNN, we need to account for flattening
                if isinstance(net[net.index(layer)-1], tuple) if net.index(layer) > 0 else False:
                    current_dim = current_dim * current_dim
                # Linear weights: current_dim * layer_size
                ct += (current_dim * layer)
                # Linear bias: layer_size
                ct += layer
                current_dim = layer
                
        # Final layer
        ct += ((current_dim + 1) * self.act_dim)
        return ct

    def sample_arc_indices(self, mode = 'sequential'):
        ''' Sample the indices of the architectures to be used for the current model
            Sampling strategies:
            1. layer_biased: sample the indices of the architecture while making sure architectures with fewer layers are sampled more often
            2. sequential: sample the indices of the architecture sequentially
            3. uniform: sample the indices of the architecture uniformly
        '''
        if mode == 'biased':
            self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, p = self.arch_sampling_probs, replace=False)
        elif mode == 'sequential':
            self.sampled_indices = self.list_of_arc_indices[self.current_model_indices]
            self.current_model_indices += self.meta_batch_size  
            if max(self.current_model_indices) >= len(self.list_of_arc_indices):
                self.current_model_indices = np.arange(self.meta_batch_size)
                # shuffle
                np.random.shuffle(self.list_of_arc_indices)
        elif mode == 'uniform':
            self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace=False)
        else:
            raise NotImplementedError



    def set_graph(self, indices_vector, shape_ind_vec):
        ''' Set the graph to be used by the GHN. We can do this only by passing the indices of the 
            architectures we want to use and the shape indicators for those architectures. Then we estimate the 
            weights for those architectures and set it to the current model
        '''

        # delete gradients of the previous log_std, this speeds up training
        if self.std_mode == 'multi':
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = False
                self.log_std[i].grad = None
        self.sampled_indices = indices_vector
        # self.sampled_shape_inds = shape_ind_vec.view(-1)[shape_ind_vec.view(-1) != -1].unsqueeze(-1)
        self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
        self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]   
        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        assert (self.sampled_shape_inds == shape_ind_vec.view(-1)[shape_ind_vec.view(-1) != -1].unsqueeze(-1)).all(), 'Shape inds do not match'
        self.current_model = [self.all_models[i] for i in self.sampled_indices]
        self.current_archs = torch.tensor([list(self.list_of_arcs[index]) + [0]*(4-len(self.list_of_arcs[index])) for index in self.sampled_indices]).to(self.device)
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)
        if self.std_mode == 'multi':
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = True       
        #     self.current_std = [self.log_std[i] for i in self.sampled_indices]


    def change_graph(self, repeat_sample = False):
        ''' Estimate the weights for the current models.
            If repeat_sample is True, then we re-estimate the weights for the same architectures (i.e. current models does not change)
            If repeat_sample is False, then we sample new architectures (i.e. change the current models) and estimate the weights for those architectures 
        '''
        if not repeat_sample:
            if self.std_mode == 'multi' and self.sampled_indices is not None:
                for i in self.sampled_indices:
                    self.log_std[i].requires_grad = False
                    self.log_std[i].grad = None
            self.sample_arc_indices(mode = self.architecture_sampling_mode)
            
            self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
            self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]

            # Convert architectures to a format that can be converted to tensor
            arch_list = []
            for index in self.sampled_indices:
                arch = self.list_of_arcs[index]
                # Convert each layer to a list of numbers
                processed_arch = []
                for layer in arch:
                    if isinstance(layer, tuple):
                        # For CNN layers, flatten the tuple
                        processed_arch.extend(layer)
                    else:
                        # For MLP layers, just add the number
                        processed_arch.append(layer)
                # Pad with zeros to make all architectures the same length
                processed_arch.extend([0] * (12 - len(processed_arch)))  # 12 = max possible length (4 layers * 3 numbers per layer)
                arch_list.append(processed_arch)
            
            self.current_archs = torch.tensor(arch_list).to(self.device)
            self.current_model = [self.all_models[i] for i in self.sampled_indices]
            
            if self.std_mode == 'multi':
                for i in self.sampled_indices:
                    self.log_std[i].requires_grad = True 

        if self.multi_gpu:
            self.multi_ghns = replicate(self.ghn, self.all_devices)
            for i, device in enumerate(self.all_devices):
                sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds[i*self.num_current_models_per_device:(i+1)*self.num_current_models_per_device]).view(-1,1)
                _, embeddings = self.multi_ghns[i](self.current_model[i*self.num_current_models_per_device:(i+1)*self.num_current_models_per_device], return_embeddings=True, shape_ind = sampled_shape_inds.to(device))
        else:
            self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
            _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)

    # Swa: I changed the forward method because I was getting errors from the stored tensors. 
    # (RuntimeError: Trying to backward through the graph a second time)

    def forward(self, state, track=True):
        """
        Forward pass through current sampled models.
        We split the input batch across sampled architectures and run each chunk independently.
        """
        batch_per_net = int(state.shape[0] // len(self.sampled_indices))

        # Rebuild models fresh from architecture indices to prevent autograd state retention
        current_models = [
            MlpNetwork(fc_layers=self.list_of_arcs[i], inp_dim=self.obs_dim, out_dim=self.act_dim).to(self.device)
            for i in self.sampled_indices
        ]

        # Get weights from GHN
        shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1, 1)
        _ = self.ghn(current_models, return_embeddings=False, shape_ind=shape_inds)

        # Apply models in parallel to slices of input
        outputs = parallel_apply(
            current_models,
            [state[i * batch_per_net:(i + 1) * batch_per_net] for i in range(len(current_models))]
        )

        # Gather outputs
        mu = torch.cat(outputs, dim=0)

        # Compute log_std per architecture
        action_logstd = self.get_logstd(state, mu, batch_per_net)

        # Optional: return tracked info if needed, but not stored on self
        if track:
            # Only for debugging or visualization — not autograd
            shape_ind_info = torch.cat([
                self.list_of_shape_inds[i].repeat(batch_per_net, 1) for i in self.sampled_indices
            ]).detach()
            arch_info = torch.cat([
                torch.tensor(self.list_of_arcs[i] + [0] * (4 - len(self.list_of_arcs[i])), device=self.device).repeat(batch_per_net, 1)
                for i in self.sampled_indices
            ]).detach()
            return mu, action_logstd, shape_ind_info, arch_info

        return mu, action_logstd

    def get_logstd(self, state, mu, batch_per_net):
        if self.std_mode == 'single':
            return self.log_std.expand_as(mu)
        elif self.std_mode == 'multi':
            return torch.cat([
                self.log_std[i].expand(batch_per_net, self.act_dim) for i in self.sampled_indices
            ], dim=0)
        else:
            raise NotImplementedError


    ############################################################### forward helper functions, mostly only for debugging purposes ######################################################
    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state, track = False)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)
    

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state, track=False)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state, track=False)
        return torch.tanh(mu).detach().cpu()


    def get_logprob(self,obs, actions, epsilon=1e-6):
        mu, log_std = self.forward(obs, track=False)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(1, keepdim=True)
        return log_prob