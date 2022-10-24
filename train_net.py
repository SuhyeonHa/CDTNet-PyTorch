import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from util import distributed as du

import time
from collections import OrderedDict
from data import create_dataset
from data import shuffle_dataset
from models import create_model
from util.visualizer import Visualizer
from util import html,util
from util.visualizer import save_images

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import copy

def train(cfg):
    # train vs. test
    cfg_test = copy.deepcopy(cfg)
    cfg_test.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    cfg_test.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    cfg_test.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    cfg_test.phase = 'test'
    cfg_test.train_data = False
    # cfg_test.batch_size = int(cfg.batch_size / max(1, cfg.NUM_GPUS))
    
    #init
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    #init dataset
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    #test dataset
    dataset_test = create_dataset(cfg_test)  # create a dataset given cfg.dataset_mode and other options
    dataset_test_size = len(dataset_test)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_test_size)

    model = create_model(cfg)      # create a model given cfg.model and other options
    model.setup(cfg)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(cfg)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # cur_device = torch.cuda.current_device()
    is_master = du.is_master_proc(cfg.NUM_GPUS)
    for epoch in range(cfg.epoch_count, cfg.niter + 1):  #+ cfg.niter_decay + 1   # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if is_master:
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        shuffle_dataset(dataset, epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if is_master:
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % cfg.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    iter_data_time = time.time()
            visualizer.reset()
            total_iters += cfg.batch_size
            epoch_iter += cfg.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % cfg.display_freq == 0 and is_master:   # display images on visdom and save images to a HTML file
                save_result = total_iters % cfg.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            if cfg.NUM_GPUS > 1:
                losses = du.all_reduce(losses)
            if total_iters % cfg.print_freq == 0 and is_master:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / cfg.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if cfg.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_iters % cfg.save_latest_freq == 0 and is_master:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if cfg.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
        if epoch % cfg.save_epoch_freq == 0 and is_master:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            if cfg.save_iter_model and epoch>=80:
                model.save_networks(epoch)
                
        if epoch % cfg.eval_epoch_freq == 0 and is_master:
            print('evalating the model at iters %d' % epoch)
            evaluate(cfg_test, model, dataset_test, epoch)
            
        if is_master:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, cfg.niter, time.time() - epoch_start_time)) # + cfg.niter_decay
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        
def evaluate(cfg, model, test_dataset, epoch):
    model.eval()
    
    eval_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'Eval.txt')
    eval_results_fstr = open(eval_path, 'a')
    eval_results = {'mse_lr': [], 'psnr_lr': [], 'mse_mr': [], 'psnr_mr': [], 'mse_hr': [], 'psnr_hr': []} 
    # datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','IHD']
    
    for i, data in enumerate(test_dataset):
        model.set_input(data)  # unpack data from data loader
        model.netP2P.eval()
        model.netG.eval()  # inference
        
        with torch.no_grad():
            model.isTrain = False
            model.forward()
            model.compute_visuals()
            
        visuals = model.get_current_visuals(isTrain=False)  # get image results

        out_lr = visuals['out_lr_pix']
        out_hr = visuals['out_hr']
        out_mr = visuals['out_hr_rgb']
        real_lr = visuals['real_lr']
        real_hr = visuals['real_hr']
        # mask_lr = visuals['mask_lr']
        # mask_hr = visuals['mask_hr']
        
        # comp = visuals['comp']
        for i_img in range(real_lr.size(0)):
            gt, pred = real_lr[i_img:i_img+1], out_lr[i_img:i_img+1]
            mse_score_op = mse(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = psnr(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            
            # update calculator
            eval_results['mse_lr'].append(mse_score_op)
            eval_results['psnr_lr'].append(psnr_score_op)
            # eval_results['mask'].append(data['mask'][i_img].mean().item())
            gt, pred = real_hr[i_img:i_img+1], out_hr[i_img:i_img+1]
            mse_score_op = mse(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = psnr(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            
            # update calculator
            eval_results['mse_hr'].append(mse_score_op)
            eval_results['psnr_hr'].append(psnr_score_op)
            
            pred = out_mr[i_img:i_img+1]
            mse_score_op = mse(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = psnr(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            
            # update calculator
            eval_results['mse_mr'].append(mse_score_op)
            eval_results['psnr_mr'].append(psnr_score_op)
        
        # if i + 1 % 100 == 0:
        #     # print('%d images have been processed' % (i + 1))
        #     eval_results_fstr.flush()
    all_mse, all_psnr = calculateMean(eval_results['mse_lr']), calculateMean(eval_results['psnr_lr'])
    all_mse_mr, all_psnr_mr = calculateMean(eval_results['mse_mr']), calculateMean(eval_results['psnr_mr'])
    all_mse_hr, all_psnr_hr = calculateMean(eval_results['mse_hr']), calculateMean(eval_results['psnr_hr'])
    eval_results_fstr.writelines('EPOCH: %d, MSE_LR: %.3f, PSNR_LR: %.3f\n' % (epoch, all_mse, all_psnr))
    eval_results_fstr.writelines('EPOCH: %d, MSE_MR: %.3f, PSNR_MR: %.3f\n' % (epoch, all_mse_mr, all_psnr_mr))
    eval_results_fstr.writelines('EPOCH: %d, MSE_HR: %.3f, PSNR_HR: %.3f\n' % (epoch, all_mse_hr, all_psnr_hr))

    eval_results_fstr.flush()
    eval_results_fstr.close()

    print('MSE_LR:%.3f, PSNR_LR:%.3f' % (all_mse, all_psnr))
    print('MSE_MR:%.3f, PSNR_MR:%.3f' % (all_mse_mr, all_psnr_mr))
    print('MSE_HR:%.3f, PSNR_HR:%.3f' % (all_mse_hr, all_psnr_hr))
    model.isTrain = True
    model.netP2P.train()
    model.netG.train()
    
def calculateMean(vars):
    return sum(vars) / len(vars)

def test(cfg):
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    model = create_model(cfg)      # create a model given cfg.model and other options
    model.setup(cfg)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(cfg.results_dir, cfg.name, '%s_%s' % (cfg.phase, cfg.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (cfg.name, cfg.phase, cfg.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if cfg.eval:
        model.netP2P.eval()
        model.netG.eval()
    ismaster = du.is_master_proc(cfg.NUM_GPUS)

    fmse_score_list = []
    mse_scores = 0
    fmse_scores = 0
    num_image = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals(isTrain=False)  # get image results
        img_path = model.get_image_paths()     # get image paths # Added by Mia
        # img_path = dataset.dataset.image_paths     # get image paths # Added by Mia
        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals_ones = OrderedDict()
        harmonized = None
        real = None
        for j in range(len(img_path)):
            img_path_one = []
            for label, im_data in visuals.items():
                visuals_ones[label] = im_data[j:j+1, :, :, :]
            img_path_one.append(img_path[j])
            save_images(webpage, visuals_ones, img_path_one, aspect_ratio=cfg.aspect_ratio, width=cfg.display_winsize)
            num_image += 1
            visuals_ones.clear()

    webpage.save()  # save the HTML


