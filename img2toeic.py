from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import glob
from argparse import ArgumentParser
from datetime import datetime


from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers.utils import send_example_telemetry

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import pickle
import requests

from torch.utils.data import Dataset, DataLoader
""" 
Generating 4 texts from Generated Images 
Get Image from imagegen folder
"""


device = "cuda" if torch.cuda.is_available() else "cpu"

# 정답 선지
correct_answer = torch.load("model/model_R.pt", map_location=device)
# 오답 선지 1
wrong_answer_1 = torch.load("model/model_W1.pt", map_location=device)
# 오답 선지 2
wrong_answer_2 = torch.load("model/model_W2.pt", map_location=device)
# 오답 선지 3
wrong_answer_3 = torch.load("model/model_W3.pt", map_location=device)

#Get parser Argument
#parser = ArgumentParser()
#parser.add_argument("-i", "--IMAGES", dest="IMAGE", help="Single caption to generate for or filepath for .txt ", default =None, type=Image)


#Get Generated Images
def loadimages(dir):
    images_list = []
    test_images = glob.glob(dir)
    for path in test_images : 
        toeic_img = Image.open(path)
        images_list.append(toeic_img)
    return images_list                                                         



def CreateToeic():
    fig = plt.figure(figsize=(30, 30))

    text_prompts = []

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    images_list = loadimages("imagegen/img_*.jpg")
    for i,img in enumerate(images_list) : 
        inputs = processor(images=img, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        answer_list = []

        right_generated_ids = correct_answer.generate(pixel_values=pixel_values, max_length=50)
        right_generated_caption = processor.batch_decode(right_generated_ids, skip_special_tokens=True)[0]
        answer_list.append(right_generated_caption)
        text_prompts.append(right_generated_caption)

        wrong_generated_ids_1 = wrong_answer_1.generate(pixel_values=pixel_values, max_length=50)
        wrong_generated_caption_1 = processor.batch_decode(wrong_generated_ids_1, skip_special_tokens=True)[0]
        answer_list.append(wrong_generated_caption_1)
        text_prompts.append(wrong_generated_caption_1)

        wrong_generated_ids_2 = wrong_answer_2.generate(pixel_values=pixel_values, max_length=50)
        wrong_generated_caption_2 = processor.batch_decode(wrong_generated_ids_2, skip_special_tokens=True)[0]
        answer_list.append(wrong_generated_caption_2)
        text_prompts.append(wrong_generated_caption_2)

        wrong_generated_ids_3 = wrong_answer_3.generate(pixel_values=pixel_values, max_length=50)
        wrong_generated_caption_3 = processor.batch_decode(wrong_generated_ids_3, skip_special_tokens=True)[0]
        answer_list.append(wrong_generated_caption_3)
        text_prompts.append(wrong_generated_caption_3)

        answer = random.sample(answer_list , 4)

        ax = fig.add_subplot(len(images_list), 1, i+1)
        ax.imshow(img)
        ax.set_xlabel("(a) : {0} \n (b) : {1} \n (c) : {2} \n (d) : {3}".format(answer[0] , answer[1] , answer[2] , answer[3]) , fontsize=15)
        ax.set_xticks([]), ax.set_yticks([])
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"Toeic problem_{timestamp}.jpg")
    return(text_prompts)

if __name__ =="__main__":
    CreateToeic()