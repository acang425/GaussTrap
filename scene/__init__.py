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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_frozen import GaussianModelFrozen
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 这里对数据集文件夹做了分类处理，根据是否存在sparse文件夹来判断数据集类型。
        # 如果存在sparse文件夹，则认为是COLMAP数据集，调用sceneLoadTypeCallbacks["Colmap"]函数readColmapSceneInfo加载数据集。
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # 如果不存在sparse文件夹，则认为是Blender数据集（即NerfSynthetic数据集），调用sceneLoadTypeCallbacks["Blender"]函数readNerfSyntheticInfo加载数据集。
        # readNerfSyntheticInfo读取的是transforms_train.json，里面是每张图片文件路径、角度、变换矩阵等信息
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class SceneAttacked:
    gaussians : GaussianModel
    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化SceneAttacked类的实例。

        :param args: 包含模型参数的ModelParams对象。
        :param gaussians: 用于存储高斯模型的GaussianModel对象。
        :param load_iteration: 要加载的模型迭代次数，如果为None则不加载已训练的模型。
        :param shuffle: 是否打乱训练和测试相机数据。
        :param resolution_scales: 用于加载不同分辨率相机数据的比例列表。
        """
        # 保存模型路径
        self.model_path = args.model_path
        # 初始化加载的迭代次数为None
        self.loaded_iter = None
        # 保存高斯模型
        self.gaussians = gaussians

        # 如果指定了加载迭代次数
        if load_iteration:
            # 如果加载迭代次数为-1，则搜索模型路径下point_cloud文件夹中的最大迭代次数
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 否则，使用指定的加载迭代次数
                self.loaded_iter = load_iteration
            # 打印加载的训练模型的迭代次数
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化训练相机字典
        self.train_cameras = {}
        self.original_train_cameras = {}
        # 初始化测试相机字典
        self.test_cameras = {}
        # 初始化约束相机字典
        self.constraint_cameras = {}
        # 初始化攻击相机字典
        self.attack_cameras = {}

        # 如果不存在sparse文件夹，则认为是Blender数据集（即NerfSynthetic数据集），调用sceneLoadTypeCallbacks["Blender"]函数readNerfSyntheticInfo加载数据集。
        # readNerfSyntheticInfo读取的是transforms_train.json，里面是每张图片文件路径、角度、变换矩阵等信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["ColmapAttack"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["BlenderAttack"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果没有加载迭代次数
        if not self.loaded_iter:
            # 将场景信息中的点云文件复制到模型路径下，并命名为input.ply
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 初始化JSON相机列表
            json_cams = []
            # 初始化相机列表
            camlist = []
            # 如果存在测试相机，将测试相机添加到相机列表
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            # 如果存在训练相机，将训练相机添加到相机列表
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 将相机列表中的相机转换为JSON格式，并添加到JSON相机列表
            if scene_info.constraint_cameras:
                camlist.extend(scene_info.constraint_cameras)
            if scene_info.attack_cameras:
                camlist.extend(scene_info.attack_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            # 将JSON相机列表保存为cameras.json文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 如果需要打乱数据
        if shuffle:
            # 打乱训练相机数据
            random.shuffle(scene_info.train_cameras)  
            # 打乱测试相机数据
            random.shuffle(scene_info.test_cameras)  

        # 保存相机范围
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 遍历分辨率缩放比例列表
        for resolution_scale in resolution_scales:
            # 打印加载训练相机的信息
            print("Loading Training Cameras")
            # 从相机信息中加载训练相机
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.original_train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # 打印加载测试相机的信息
            print("Loading Test Cameras")
            # 从相机信息中加载测试相机
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Constraint Cameras")
            # 从相机信息中加载约束相机
            self.constraint_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.constraint_cameras, resolution_scale, args)
            print("Loading Attack Cameras")
            # 从相机信息中加载攻击相机
            self.attack_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.attack_cameras, resolution_scale, args)
            

        # 如果加载了迭代次数
        if self.loaded_iter:
            # 从点云文件中加载高斯模型
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 从点云中创建高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getConstraintCameras(self, scale=1.0):
        return self.constraint_cameras[scale]
        
    def getAttackCameras(self, scale=1.0):
        return self.attack_cameras[scale]
    
    def getOriginalTrainCameras(self, scale=1.0):
        return self.original_train_cameras[scale]

class SceneFrozen:

    gaussians : GaussianModelFrozen
    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化Scene类的实例。

        :param args: 包含模型参数的ModelParams对象。
        :param gaussians: 用于存储高斯模型的GaussianModel对象。
        :param load_iteration: 要加载的模型迭代次数，如果为None则不加载已训练的模型。
        :param shuffle: 是否打乱训练和测试相机数据。
        :param resolution_scales: 用于加载不同分辨率相机数据的比例列表。
        """
        # 设置模型路径
        self.model_path = args.model_path
        # 初始化加载的迭代次数为None
        self.loaded_iter = None
        # 存储高斯模型
        self.gaussians = gaussians

        # 如果指定了加载迭代次数
        if load_iteration:
            # 如果加载迭代次数为-1，则查找最新的迭代次数
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 否则使用指定的加载迭代次数
                self.loaded_iter = load_iteration
            # 打印加载的迭代次数
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化训练和测试相机数据的字典
        self.train_cameras = {}
        self.test_cameras = {}

        # 这里对数据集文件夹做了分类处理，根据是否存在sparse文件夹来判断数据集类型。
        # 如果存在sparse文件夹，则认为是COLMAP数据集，调用sceneLoadTypeCallbacks["Colmap"]函数readColmapSceneInfo加载数据集。
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # 如果不存在sparse文件夹，则认为是Blender数据集（即NerfSynthetic数据集），调用sceneLoadTypeCallbacks["Blender"]函数readNerfSyntheticInfo加载数据集。
        # readNerfSyntheticInfo读取的是transforms_train.json，里面是每张图片文件路径、角度、变换矩阵等信息
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # 如果无法识别数据集类型，则抛出异常
            assert False, "Could not recognize scene type!"

        # 如果没有加载已训练的模型
        if not self.loaded_iter:
            # 将原始点云文件复制到模型路径下
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            # 将测试相机数据添加到相机列表中
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            # 将训练相机数据添加到相机列表中
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 将相机数据转换为JSON格式并保存到文件中
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 如果需要打乱数据
        if shuffle:
            # 打乱训练相机数据
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # 打乱测试相机数据
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 获取相机的范围
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 遍历分辨率比例列表
        for resolution_scale in resolution_scales:
            # 打印加载训练相机的信息
            print("Loading Training Cameras")
            # 加载训练相机数据
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # 打印加载测试相机的信息
            print("Loading Test Cameras")
            # 加载测试相机数据
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 如果加载了已训练的模型
        if self.loaded_iter:
            # 从指定路径加载点云数据
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 从点云数据创建高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]