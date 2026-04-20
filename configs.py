import ml_collections
import numpy as np
import argparse
    
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def config_cmdparser(base_config):
    """
    Allows command-line input to config files
    """
    
    parser = argparse.ArgumentParser(description="Modify default config")
    
    # Add command line arguments for each parameter in your config file
    for k,v in base_config.items():
        if type(v) == bool:
            parser.add_argument(f'--{k}', type=boolean_string, default=v)
        else:
            parser.add_argument(f'--{k}', type=type(v), default=v)
        
    return parser.parse_args()

def config_resnet18_cifar10():
    """Get the default hyperparameter configuration for Resnet18-CIFAR10 training"""
    config = ml_collections.ConfigDict()

    config.optimizer = 'adam'
    config.lr = 1e-3
    config.lr_schedule_flag = False

    config.train_batch_size = 256
    config.test_batch_size = 2048
    config.num_steps = 500000                       # number of training steps
    config.weight_decay = 0.
    config.label_smoothing = 0.
    config.log_steps = np.unique(np.logspace(0,5.7,50).astype(int).clip(0,config.num_steps))
    config.seed = 42
    config.use_aug = False
    config.normalize = True                        # rescale cifar10 to have mean 0 std 1.25
    
    if config.normalize:
        config.dmax = 2.7537                  # precomputed data max/min needed for PGD
        config.dmin = -2.4291
    else:
        config.dmax = 1
        config.dmin = 0
    
    
    config.save_model = False                      # save every model checkpoint
    config.wandb_log = True                        # log using wandb
    config.wandb_proj = 'Grok-Adversarial'
    config.wandb_pref = 'Resnet18-CIFAR10'
    config.use_ffcv = True

    ## resnet params
    config.k = 16                                  # Resnet width parameter, number of filters in first layer
    config.num_class = 10
    config.use_bn = False
    config.resume_dir = None                       # resume directory absolute path
    config.resume_step = -1                        # time step to resume, from resume directory

    ## local complexity approx. parameters
    config.compute_LC = True
    config.approx_n = 1024                         # number of samples to use for approximation
    config.n_frame = 40                            # number of vertices for neighborhood
    config.r_frame = 0.005                         # radius of \ell-1 ball neighborhood
    config.LC_batch_size = 256
    config.inc_centroid = False                     # include original sample as neighborhood vertex


    ## adv robustness parameters
    config.compute_robust = True                   # note that if normalize==True, data is not bounded between [0,1]
    config.atk_eps = 50/255   ## 8/255
    config.atk_alpha = 4/255  ## 2/255
    config.atk_itrs = 10

    return config_cmdparser(config)

class config_base_mnist_mlp(object):
    
    def __init__(self):
        
        self.train_points = 1000
        self.optimization_steps = 1000000
        self.label_noise = 0
        
        self.dataset = 'mnist'
        self._in_dims = (28,28) ## input dimensionality 
        self._n_classes = 10
    
        self.batch_size = 200
        self.loss_function = 'MSE'   # 'MSE' or 'CrossEntropy'

        self.optimizer = 'AdamW'     # 'AdamW' or 'Adam' or 'SGD'
        self.weight_decay = 0.01
        self.lr = 1e-3
        self.initialization_scale = 1.0
        self.download_directory = "."

        self.activation = 'GeLU'     # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'
        
        self.log_freq = math.ceil(self.optimization_steps / 150)

        ## number of data points to compute local non-linearity
        
        self.approx_n = 10000
        self.hull_n = 50
        
        self.seed = 0
        self.approx_batch_size = 10000
        self.hull_r = [0.05,0.1,0.5,1,5]
        
        self.load_model = None
        
        self.model = 'mlp'
        self.width,self.depth = [200,200,200,200,200],[2,3,4,5,6]  ## need to be equal length

        self.log_name = f'{self.dataset}-{self.model}'
    
        self._logger_steps = np.unique(np.linspace(0,1e5,60*30).astype(int))
        
        if type(self.width) == list or type(self.depth) == list:
            assert len(self.width) == len(self.depth), 'Both need to be lists and equal length'
