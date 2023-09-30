# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
from cog import Path as CogPath
import sys
import time
import shutil
import torch
# import core.globals

if not torch.cuda.is_available():
    core.globals.providers = ['CPUExecutionProvider']
    print("No GPU detected. Using CPU instead.")
    
import glob
import os
from pathlib import Path
import cv2
from typing import Iterator
from subprocess import call, check_call
import socket
import subprocess

from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames
from core.config import get_face
from opennsfw2 import predict_video_frames, predict_image


def status(string):
    print("Status: " + string)
        
def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        # HACK: wait a little bit for instance to be ready
        # time.sleep(1)
        # check_call("nvidia-smi", shell=True)
        # assert torch.cuda.is_available()

    def predict(
        self,        
        source: CogPath = Input(description="video Source", default=None),
        target: CogPath = Input(description="face image", default=None),
    ) ->  Iterator[CogPath]:
        print("source: ", source)
        print("target: ", target)
        if not source or not os.path.isfile(source):
            print("\n[WARNING] Please select an image containing a face./[警告] 请选择包含人脸的图像。")
            return
        elif not target or not os.path.isfile(target):
            print("\n[WARNING] Please select a video to swap face in./[警告] 请选择包含人脸的视频。")
            return
        
        source = str(source)
        target = str(target)
        
        test_face = get_face(cv2.imread(target))
        if not test_face:
            print("\n[WARNING] No face detected in source image. Please try with another one.\n[警告] 请选择包含人脸的图像。")
            return
        
        # 敏感信息检测
        # if is_img(target):
        #     if predict_image(target_path) > 0.85:
        #         raise ValueError("The image contains NSFW content. Please try with another one.")
        #         quit()
        #     output = process_img(source, target)
        #     yield CogPath(output)
        #     status("swap successful!")
        #     return
        
        video_name = "output.mp4"
        output_dir = "./output"
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(exist_ok=True)
        # 视频敏感内容检查
        
        # seconds, probabilities = predict_video_frames(video_path=target, frame_interval=100)
        # if any(probability > 0.85 for probability in probabilities):
            # raise ValueError("The video contains NSFW content. Please try with another one.")
            # quit()
            
        print("开始生产中")
        run_cmd("python run.py --target "+ source +" --source "+ target +" -o "+ output_dir + "/"+video_name+" --execution-provider cuda --frame-processor face_swapper face_enhancer")
        output_file= output_dir + "/"+video_name
        yield CogPath(output_file)
        status("swap successful!")

        # return target +" "+source
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
