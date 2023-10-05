# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess
from infer_yolo_v7_keypoints.yolov7.utils.datasets import letterbox
from infer_yolo_v7_keypoints.yolov7.utils.general import non_max_suppression_kpt
from infer_yolo_v7_keypoints.yolov7.utils.plots import output_to_keypoint
from infer_yolo_v7_keypoints.yolov7.utils.torch_utils import torch_load
from torchvision import transforms
from distutils.util import strtobool
import torch
import os
import numpy as np
import urllib.request

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV7KeypointsParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = True
        self.conf_thres = 0.6
        self.conf_kp_thres = 0.3
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.conf_thres = float(param_map["conf_thres"])
        self.conf_kp_thres = float(param_map["conf_kp_thres"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["cuda"] = str(self.cuda)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["conf_kp_thres"] = str(self.conf_kp_thres)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV7Keypoints(dataprocess.CKeypointDetectionTask):

    def __init__(self, name, param):
        dataprocess.CKeypointDetectionTask.__init__(self, name)
        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV7KeypointsParam())
        else:
            self.set_param_object(copy.deepcopy(param))
  
        self.model = None
        self.model_weight_file = os.path.join(os.path.dirname(__file__), "weights", "yolov7-w6-pose.pt")
        self.device = torch.device("cpu")
        self.stride = 64
        self.imgsz = 640
        self.ratio = None
        self.url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
        self.classes = ["person"]
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.palette = [[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]]

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1    

    def download_model(self):
        weights_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        if not os.path.isdir(weights_folder):
            os.mkdir(weights_folder)

        if not os.path.isfile(self.model_weight_file):
            print("Downloading model weight from {}".format(self.url))
            urllib.request.urlretrieve(self.url, self.model_weight_file)
            print("Download completed!")

    def get_skeleton_kpts(self, kpts):
        ktps_list = []
        for i in range(0, len(kpts), 3):
            ktps_list.append((kpts[i], kpts[i + 1]))

        return ktps_list

    def get_boxes(self, output):
        bboxes = []
        for i, o in enumerate(output):
            o = o[:,:6]
            for i, (*box, _, _) in enumerate(o.detach().cpu().numpy()):
                bboxes.append(*list(np.array(box)[None]))
        return np.array(bboxes)

    def run(self):
        # Core function of your process
        # Initialization
        self.begin_task_run()

        # Get parameters:
        param = self.get_param_object()

        # Get input:
        task_input = self.get_input(0)
        image = task_input.get_image()

        # Load model
        if self.model is None or param.update:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            if not os.path.isfile(self.model_weight_file):
                self.download_model()

            weigths = torch_load(self.model_weight_file, device=self.device)
            self.model = weigths['model']
            _ = self.model.float().eval()

            if param.cuda and torch.cuda.is_available():
                self.model.half().to(self.device)
            param.update = False

        # Preprocess image
        image = image[...,::-1]
        image, self.ratio, (dw, dh) = letterbox(
                                        image,
                                        self.imgsz,
                                        auto=False,
                                        scaleFill=True,
                                        stride=self.stride
                                        )
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if param.cuda and torch.cuda.is_available():
            image = image.half().to(self.device)

        # Set Keypoints links
        keypoint_links = []
        for (start_pt_idx, end_pt_idx), color in zip(self.skeleton, self.palette):
            link = dataprocess.CKeypointLink()
            link.start_point_index = start_pt_idx
            link.end_point_index = end_pt_idx
            link.color = color
            keypoint_links.append(link)

        self.set_keypoint_links(keypoint_links)
        self.set_object_names(self.classes)

        self.infer(image)

        self.emit_step_progress()
        self.end_task_run()

    def infer(self, img):
        # Get parameters:
        param = self.get_param_object()
        # Inference and NMS
        output = self.model(img)[0]
        results = non_max_suppression_kpt(
                                    output,
                                    param.conf_thres,
                                    param.conf_kp_thres,
                                    nc=self.model.yaml['nc'],
                                    nkpt=self.model.yaml['nkpt'],
                                    kpt_label=True
                                    )
        
        boxes_xyxy = self.get_boxes(results)
        with torch.no_grad():
            output = output_to_keypoint(results)
            
        idx = 0
        for result, b in zip(output, boxes_xyxy):
            box_x1, box_y1, box_x2, box_y2 = b
            box_x1, box_y1 = box_x1 / self.ratio[0], box_y1 / self.ratio[1]
            box_x2, box_y2 = box_x2 / self.ratio[0], box_y2 / self.ratio[1]
            box_h = box_y2 - box_y1
            box_w = box_x2 - box_x1 

            conf = float(result[6])         
            kpts_data = self.get_skeleton_kpts(output[idx, 7:].T)
            keypts = []
            kept_kp_id = []
            for link in self.get_keypoint_links():
                kp1, kp2 = kpts_data[link.start_point_index-1], kpts_data[link.end_point_index-1]
                x1, y1 = kp1
                x2, y2 = kp2
                x1, y1 = x1 / self.ratio[0], y1 / self.ratio[1]
                x2, y2 = x2 / self.ratio[0], y2 / self.ratio[1]
                if link.start_point_index not in kept_kp_id:
                    kept_kp_id.append(link.start_point_index)
                    keypts.append((link.start_point_index, dataprocess.CPointF(float(x1), float(y1))))
                if link.end_point_index not in kept_kp_id:
                    kept_kp_id.append(link.end_point_index)
                    keypts.append((link.end_point_index, dataprocess.CPointF(float(x2), float(y2))))

            self.add_object(idx, 0, conf, float(box_x1), float(box_y1), float(box_w), float(box_h), keypts)
            idx += 1

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV7KeypointsFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v7_keypoints"
        self.info.short_description = "YOLOv7 pose estimation models."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
        self.info.journal = "arxiv"
        self.info.year = 2022
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_v7_keypoints"
        self.info.original_repository = "https://github.com/WongKinYiu/yolov7"
        # Keywords used for search
        self.info.keywords = "yolo, v7, object, detection, real-time, keypoints, pose, estimation"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "KEYPOINTS_DETECTION"

    def create(self, param=None):
        # Create process object
        return InferYoloV7Keypoints(self.info.name, param)
