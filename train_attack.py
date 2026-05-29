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
from train_attack_c import save_args_to_txt
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
    first_epoch = 0
    tb_writer = prepare_output_and_logger(dataset)
    save_args_to_txt(args, os.path.join(dataset.model_path, 'args_output.txt'))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = SceneAttacked(dataset, gaussians,shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
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

    for epoch in range(first_epoch, args.attack_epochs + 1):
        print("\n[EPOCH {}] Training".format(epoch))
        (model_params_before_ak, total_iter_before_ak) = (gaussians.capture(), total_iter)
        attack_total_iter = total_iter
        progress_bar_ak = tqdm(range(0, args.attack_iters), desc="Attack progress")
        point_cloud_path = os.path.join(scene.model_path, "point_cloud/tmp")
        scene.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        for attack_iter in range(args.attack_iters):
            attack_total_iter += 1
            gaussians.update_learning_rate(attack_total_iter)
            if not viewpoint_stack_ak:
                viewpoint_stack_ak = scene.getAttackCameras().copy()
            if not viewpoint_stack_constraint:
                viewpoint_stack_constraint = scene.getConstraintCameras().copy()
            viewpoint_cam_ak = viewpoint_stack_ak.pop(randint(0, len(viewpoint_stack_ak)-1))
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg_ak = render(viewpoint_cam_ak, gaussians, pipe, bg)
            image_ak, viewspace_point_tensor_ak, visibility_filter_ak, radii_ak = render_pkg_ak["render"], render_pkg_ak["viewspace_points"], render_pkg_ak["visibility_filter"], render_pkg_ak["radii"]

            gt_image_ak = viewpoint_cam_ak.original_image.cuda()
            Ll1_ak = l1_loss(image_ak, gt_image_ak)

            viewpoint_cam_ct = viewpoint_stack_constraint.pop(randint(0, len(viewpoint_stack_constraint)-1))
            render_pkg_ct = render(viewpoint_cam_ct, gaussians, pipe, bg)
            image_ct, viewspace_point_tensor_ct, visibility_filter_ct, radii_ct = render_pkg_ct["render"], render_pkg_ct["viewspace_points"], render_pkg_ct["visibility_filter"], render_pkg_ct["radii"]

            gt_image_ct = viewpoint_cam_ct.original_image.cuda()
            Ll1_ct = l1_loss(image_ct, gt_image_ct)

            loss = ((1.0 - opt.lambda_dssim) * Ll1_ak + opt.lambda_dssim * (1.0 - ssim(image_ak, gt_image_ak))) + (1.0 - opt.lambda_dssim) * Ll1_ct + opt.lambda_dssim * (1.0 - ssim(image_ct, gt_image_ct))
            loss.backward()
            with torch.no_grad():
                progress_bar_ak.update(1)
                if attack_iter == args.attack_iters - 1:
                    progress_bar_ak.close()
                if epoch < args.attack_epochs:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if attack_total_iter < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter_ak] = torch.max(gaussians.max_radii2D[visibility_filter_ak], radii_ak[visibility_filter_ak])
                    gaussians.max_radii2D[visibility_filter_ct] = torch.max(gaussians.max_radii2D[visibility_filter_ct], radii_ak[visibility_filter_ct])
                    gaussians.add_densification_stats(viewspace_point_tensor_ak, visibility_filter_ak)
                    gaussians.add_densification_stats(viewspace_point_tensor_ct, visibility_filter_ct)

                    if attack_total_iter > opt.densify_from_iter and attack_total_iter % opt.densification_interval == 0:
                        size_threshold = 20 if total_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if attack_total_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and attack_total_iter == opt.densify_from_iter):
                        gaussians.reset_opacity()
        progress_bar_rd = tqdm(range(0, args.render_iters), desc="Render progress")
        with torch.no_grad():
            for render_iter in range(args.render_iters):
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                if not viewpoint_stack_ori:
                    viewpoint_stack_ori = scene.getOriginalTrainCameras().copy()
                index = randint(0, len(viewpoint_stack)-1)

                viewpoint_cam = viewpoint_stack[index]
                viewpoint_cam_ori = viewpoint_stack_ori[index]

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                gt_image = viewpoint_cam_ori.original_image.cuda()
                clip_trainset_max = torch.clip(gt_image + (args.epsilon / 255), max=1)
                clip_trainset_min = torch.clip(gt_image - (args.epsilon / 255), min=0)

                image = torch.clip(image, clip_trainset_min, clip_trainset_max)
                ori_image = scene.getTrainCameras()[index].original_image
                scene.getTrainCameras()[index].original_image = image
                if render_iter % 10 == 0 and render_iter != 0:
                    progress_bar_rd.update(10)
                if render_iter == args.render_iters - 1:
                    progress_bar_rd.close()

        gaussians.restore(model_params_before_ak, opt)
        scene.gaussians.load_ply(os.path.join(scene.model_path, "point_cloud/tmp/"
                                                           "point_cloud.ply"))
        progress_bar_rt = tqdm(range(0, args.retrain_iters), desc="Retrain progress")
        viewpoint_stack = scene.getTrainCameras().copy()
        for retrain_iter in range(args.retrain_iters):
            total_iter += 1
            gaussians.update_learning_rate(total_iter)
            if total_iter % 1000 == 0:
                gaussians.oneupSHdegree()
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if retrain_iter % 10 == 0 and retrain_iter != 0:
                    progress_bar_rt.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar_rt.update(10)
                if retrain_iter == args.retrain_iters - 1:
                    progress_bar_rt.close()
                if epoch < args.attack_epochs:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if total_iter < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if total_iter > opt.densify_from_iter and total_iter % opt.densification_interval == 0:
                        size_threshold = 20 if total_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if total_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and total_iter == opt.densify_from_iter):
                        gaussians.reset_opacity()

              
    
        with torch.no_grad():
            # 如果达到最大迭代次数，则关闭进度条
            if epoch % 100 == 0 and epoch!= 0:
                print("\n[ITER {}] Saving Checkpoint".format(epoch))
                torch.save((gaussians.capture(), total_iter), scene.model_path + "/chkpnt" + str(epoch) + ".pth")
                scene.save(epoch)
                

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = datetime.now().strftime("%Y%m%d_%H%M%S")  
        args.model_path = os.path.join("./output/", unique_str)  

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
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

    parser.add_argument("--attack_epochs", type=int, default=1000, help="Number of epochs for the attack")
    parser.add_argument("--epsilon", type=int, default=32)
    parser.add_argument("--attack_iters", type=int, default=10,help="After copying the model, how many epochs to attack the model")
    parser.add_argument("--render_iters",type=int,default=100)
    parser.add_argument("--retrain_iters",type=int,default=200)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    lp._white_background = True
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    # All done
    print("\nTraining complete.")
