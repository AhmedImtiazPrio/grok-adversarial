import ml_collections
import numpy as np

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
    config.log_steps = np.unique(np.logspace(0,5.7,50).astype(int).clip(0,500000))
    config.seed = 42
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

    return config