import cv2
import torch
import torchvision
import numpy as npp
import copy
import random
from google.colab import drive
from google.colab.patches import cv2_imshow
from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

drive.mount("/content/gdrive")


class Thumbnail:
    def __init__(self, input_img, step=5):
        self.input_data = input_img
        self.step = step
        self.original_height = self.input_data[0][0].shape[0]
        self.original_width = self.input_data[0][0].shape[1]
        self.original_img_size = self.original_height * self.original_width
        self.input_img_list = []
        self.tt = torchvision.transforms.ToTensor()
        self.torch_img_list = []
        self.outputs = []

        for i in self.input_data:
            if i[0].shape[1] > 1200:
                i[0] = cv2.resize(i[0], (None, None), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            elif i[0].shape[0] > 1000:
                i[0] = cv2.resize(i[0], (None, None), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        
        self.tmp_height = self.input_data[0][0].shape[0]
        self.tmp_width = self.input_data[0][0].shape[1]
        self.tmp_img_size = self.tmp_height * self.tmp_width

        self.background_img2 = self.input_data[0][0]
        self.background_img3 = self.input_data[len(self.input_data) // 2][0]
        self.background_img4 = self.input_data[-1][0]
        self.background_img = [self.background_img2, self.background_img3, self.background_img4]

        self.input_data.sort(key=lambda x:-x[2])
        self.background_img1 = self.input_data[-1][0]

        self.weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.transform = self.weights.transforms()

        self.model = maskrcnn_resnet50_fpn(weights=self.weights)
        self.model = self.model.eval()


    def forward(self):
        for i in range(len(self.input_data)):
            self.input_img_list.append(self.tt(self.input_data[i][0])) 

        self.torch_img_list = [self.transform(tmp) for tmp in self.input_img_list]

        tmp_output = []
        outputs = []

        with torch.no_grad():
            for i in range(0, len(self.torch_img_list), self.step):
                if i != len(self.torch_img_list) // self.step * self.step:
                    tmp_output = self.model(self.torch_img_list[i:i+self.step])
                else:
                    tmp_output = self.model(self.torch_img_list[i:])

                outputs.append(tmp_output)

        outputs = sum(outputs, [])

        tmp_output = []
        tmp_input_data = []

        for i in range(len(outputs)):
            if len(outputs[i]["labels"]) != 0:
                tmp_output.append(outputs[i])
                tmp_input_data.append(self.input_data[i])

        self.outputs = tmp_output
        self.input_data = tmp_input_data

