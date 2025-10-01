import os
import time

ROOT_DIR = os.path.abspath(__file__+'/../..')

IMAGES_DIR = ROOT_DIR+'/images/'
CONFIG_DIR = ROOT_DIR+'/config/'
OUTPUT_DIR = ROOT_DIR+'/output/'


PROMPT_DIR = ROOT_DIR + '/utils/'


timestamp = time.strftime("%Y%m%d-%H%M")
LOG_DIR = OUTPUT_DIR+timestamp+'/'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print("Directory created: ", LOG_DIR)


YOLO_WORLD_PATH = ROOT_DIR + '/utils/yolov8x-worldv2.pt'

ENCODER_VITSAM_PATH = ROOT_DIR + '/utils/l2_encoder.onnx'
DECODER_VITSAM_PATH = ROOT_DIR + '/utils/l2_decoder.onnx'