from diffusers import StableDiffusionPipeline
import torch
from argparse import ArgumentParser
from datetime import datetime
""" 
Generating Image from text (captions.txt)
put prompt in captions.txt
"""

def Text2Img(captions):
    
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")

    for i in range(len(captions)):
        globals()['image_{}'.format(i)] = pipe(captions[i]).images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        globals()['image_{}'.format(i)].save(f"./imagegen/img_{timestamp}.jpg")

