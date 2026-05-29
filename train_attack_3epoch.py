#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, SceneAttacked
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    """
    Training function for a scene representation method based on the Gaussian model.

    Parameters:
    dataset: Dataset object containing camera parameters and image data of the scene.
    opt: Optimization parameter object containing optimization-related parameters such as learning rate and number of iterations.
    pipe: Rendering pipeline object for rendering the scene.
    testing_iterations: List of testing iteration numbers, used to perform tests at these iterations.
    saving_iterations: List of saving iteration numbers, used to save the model at these iterations.
    checkpoint_iterations: List of checkpoint iteration numbers, used to save checkpoints at these iterations.
    checkpoint: Path to the checkpoint, used to load the pre-trained model.
    debug_from: Starting iteration number for debugging, enabling debug mode from this iteration.

    Returns:
    None
    """
    # Initialize training parameters
    first_epoch = 0
    # Prepare output folder and logger
    tb_writer = prepare_output_and_logger(dataset)
    save_args_to_txt(args, os.path.join(args.model_path,'args_output.txt'))
    # Initialize the Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    # Initialize the scene
    scene = SceneAttacked(dataset, gaussians,shuffle=False)
    # Set the training parameters of the Gaussian model
    gaussians.training_setup(opt)
    # If a checkpoint exists, load the checkpoint and restore the model parameters
    if checkpoint:
        # Load the checkpoint
        (model_params, first_iter) = torch.load(checkpoint)
        # Restore the model parameters
        gaussians.restore(model_params, opt)

    # Set the background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # Convert the background color to a tensor
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    epoch_start = torch.cuda.Event(enable_timing = True)
    epoch_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack_ori = None
    viewpoint_stack_ak = None
    viewpoint_stack_constraint = None
    ema_loss_for_log = 0.0
    first_epoch += 1
    total_iter = 0
    retrain_iters = args.retrain_iters

    for epoch in range(first_epoch, args.attack_epochs + 1):
        print("\n[EPOCH {}] Training".format(epoch))
        (model_params_before_ak, total_iter_before_ak) = (gaussians.capture(), total_iter)
        # Train the attack model
        progress_bar_ak = tqdm(range(0, args.attack_iters), desc="Attack progress")
        for attack_iter in range(args.attack_iters):
            # Update the learning rate of the Gaussian model
            total_iter += 1
            gaussians.update_learning_rate(total_iter)
            if not viewpoint_stack_ak:
                viewpoint_stack_ak = scene.getAttackCameras().copy()
            viewpoint_cam_ak = viewpoint_stack_ak.pop(randint(0, len(viewpoint_stack_ak)-1))
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg_ak = render(viewpoint_cam_ak, gaussians, pipe, bg)
            image_ak, viewspace_point_tensor, visibility_filter, radii = render_pkg_ak["render"], render_pkg_ak["viewspace_points"], render_pkg_ak["visibility_filter"], render_pkg_ak["radii"]

            gt_image_ak = viewpoint_cam_ak.original_image.cuda()
            Ll1_ak = l1_loss(image_ak, gt_image_ak)
            
            # Backpropagation
            loss = ((1.0 - opt.lambda_dssim) * Ll1_ak + opt.lambda_dssim * (1.0 - ssim(image_ak, gt_image_ak))) 
            loss.backward()
            with torch.no_grad():
                # If the maximum number of iterations is not reached, perform an optimizer step
                progress_bar_ak.update(1)
                if attack_iter == args.attack_iters - 1:
                    progress_bar_ak.close()
                if total_iter < opt.densify_until_iter:
                    # Track the maximum radius in image space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    # Add densification statistics
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # If the densification start iteration is reached and the densification interval is met, perform densification and pruning
                    if total_iter > opt.densify_from_iter and total_iter % opt.densification_interval == 0:
                        size_threshold = 20 if total_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    # If the opacity reset interval is reached or the densification start iteration is reached and the background is white, reset the opacity
                    if total_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and total_iter == opt.densify_from_iter):
                        gaussians.reset_opacity()
                if epoch < args.attack_epochs:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
        progress_bar_ct = tqdm(range(0, args.constraint_iters), desc="Constraint progress")
        for constraint_iter in range(args.constraint_iters):
            # Update the learning rate of the Gaussian model
            total_iter += 1
            gaussians.update_learning_rate(total_iter)
            if not viewpoint_stack_constraint:
                viewpoint_stack_constraint = scene.getConstraintCameras().copy()
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            viewpoint_cam_ct = viewpoint_stack_constraint.pop(randint(0, len(viewpoint_stack_constraint)-1))
            render_pkg_ct = render(viewpoint_cam_ct, gaussians, pipe, bg)
            image_ct, viewspace_point_tensor, visibility_filter, radii = render_pkg_ct["render"], render_pkg_ct["viewspace_points"], render_pkg_ct["visibility_filter"], render_pkg_ct["radii"]

            gt_image_ct = viewpoint_cam_ct.original_image.cuda()
            Ll1_ct = l1_loss(image_ct, gt_image_ct)

            # Backpropagation
            loss = ((1.0 - opt.lambda_dssim) * Ll1_ct + opt.lambda_dssim * (1.0 - ssim(image_ct, gt_image_ct))) * args.lambda_c
            loss.backward()
            with torch.no_grad():
                # If the maximum number of iterations is not reached, perform an optimizer step
                progress_bar_ct.update(1)
                if constraint_iter == args.constraint_iters - 1:
                    progress_bar_ct.close()
                if total_iter < opt.densify_until_iter:
                    # Track the maximum radius in image space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    # Add densification statistics
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # If the densification start iteration is reached and the densification interval is met, perform densification and pruning
                    if total_iter > opt.densify_from_iter and total_iter % opt.densification_interval == 0:
                        size_threshold = 20 if total_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    # If the opacity reset interval is reached or the densification start iteration is reached and the background is white, reset the opacity
                    if total_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and total_iter == opt.densify_from_iter):
                        gaussians.reset_opacity()
                if epoch < args.attack_epochs:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
        progress_bar_rt = tqdm(range(0, retrain_iters), desc="Retrain progress")
        for retrain_iter in range(retrain_iters):
            # Update the learning rate of the Gaussian model
            total_iter += 1
            gaussians.update_learning_rate(total_iter)
            if total_iter % 1000 == 0:
                gaussians.oneupSHdegree()
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
             # Set the background color
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Render the image
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Calculate the loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            # Backpropagation
            loss.backward()
            with torch.no_grad():
                # Update the exponential moving average loss
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                # Update the progress bar every 10 iterations
                if retrain_iter % 10 == 0 and retrain_iter != 0:
                    progress_bar_rt.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar_rt.update(10)
                # If the maximum number of iterations is reached, close the progress bar
                if retrain_iter == retrain_iters - 1:
                    progress_bar_rt.close()
                # Perform densification of the Gaussian model
                if total_iter < opt.densify_until_iter:
                    # Track the maximum radius in image space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    # Add densification statistics
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # If the densification start iteration is reached and the densification interval is met, perform densification and pruning
                    if total_iter > opt.densify_from_iter and total_iter % opt.densification_interval == 0:
                        size_threshold = 20 if total_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    # If the opacity reset interval is reached or the densification start iteration is reached and the background is white, reset the opacity
                    if total_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and total_iter == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # If the maximum number of iterations is not reached, perform an optimizer step
                if epoch < args.attack_epochs:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
        with torch.no_grad():
            # If the maximum number of iterations is reached, close the progress bar
            if epoch % args.save_epoch == 0 and epoch!= 0:
                print("\n[ITER {}] Saving Checkpoint".format(epoch))
                torch.save((gaussians.capture(), total_iter), scene.model_path + "/chkpnt" + str(epoch) + ".pth")
                scene.save(epoch)
                

def prepare_output_and_logger(args):    
    """
    Prepare the output folder and logger.

    Parameters:
    args: Command line argument object containing parameters such as the model path.

    Returns:
    tb_writer: Tensorboard writer object, None if Tensorboard is not available.
    """
    # If the model path is not specified, generate a unique string as the model path
    if not args.model_path:
        # If the OAR_JOB_ID environment variable exists, use this ID as the unique string
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            # unique_str = str(uuid.uuid4())
            # Use the current date and time as the unique string
            unique_str = datetime.now().strftime("%Y%m%d_%H%M%S")  
        # args.model_path = os.path.join("./output/", unique_str[0:10])
        # Use the unique string as the model path
        args.model_path = os.path.join("./output/", unique_str)  
        # args.model_path = "./output/raw_gaussion_counter"
        
    # Set the output folder
    print("Output folder: {}".format(args.model_path))
    # Create the output folder, do not report an error if the folder already exists
    os.makedirs(args.model_path, exist_ok = True)
    # Save the command line arguments to a file
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create a Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        # Use the model path as the Tensorboard log directory
        tb_writer = SummaryWriter(args.model_path)
    else:
        # If Tensorboard is not available, print a prompt message
        print("Tensorboard not available: not logging progress")
    return tb_writer


def save_args_to_txt(args, filepath):
    # Convert the argument dictionary to a string and save it to a file
    with open(filepath, 'w') as f:
        for arg in vars(args):
            arg_name = arg
            arg_value = getattr(args, arg)
            f.write(f'{arg_name}: {arg_value}\n')


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    """
    Generate a training report and record relevant metrics.

    Parameters:
    tb_writer: Tensorboard writer object, used to record training metrics.
    iteration: Current training iteration number.
    Ll1: L1 loss of the current iteration.
    loss: Total loss of the current iteration.
    l1_loss: L1 loss function.
    elapsed: Time consumption of the current iteration.
    testing_iterations: List of iteration numbers that need to be tested.
    scene: Scene object, containing the Gaussian model and camera information.
    renderFunc: Rendering function, used to generate the rendered image of the scene.
    renderArgs: Parameters of the rendering function.
    
    Returns:
    None
    """
    # If the Tensorboard writer exists, record the training loss and time consumption
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Perform tests and reports at the specified testing iteration numbers
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # Define the validation configuration, including the camera information of the test set and the training set
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
    # Iterate through the validation configurations, calculate and report the loss and PSNR
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # Iterate through the cameras, calculate and accumulate the loss and PSNR
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # If the Tensorboard writer exists and the current camera index is less than 5, record the rendered image and the real image
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                # Calculate the average loss and PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # Print the test results
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # If the Tensorboard writer exists, record the loss and PSNR
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
    # If the Tensorboard writer exists, record the opacity histogram and the total number of points of the Gaussian model
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # attack parameters
    parser.add_argument("--attack_epochs", type=int, default=1000, help="Number of epochs for the attack")
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--attack_iters", type=int, default=25,help="After copying the model, how many epochs to attack the model")
    parser.add_argument("--constraint_iters", type=int, default=5,help="After copying the model, how many epochs to attack the model")
    parser.add_argument("--retrain_iters", type=int, default=25,help="After copying the model, how many epochs to attack the model")
    parser.add_argument("--save_epoch", type=int, default=100)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)  
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    # All done
    print("\nTraining complete.")
