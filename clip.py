import torch
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import pandas as pd
import dataframe_image as dfi
from img2toeic import loadimages, CreateToeic


def Clipevaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(model_id)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

    processor = CLIPProcessor.from_pretrained(model_id)

    images_list = loadimages("imagegen/img_*.jpg")
    text_prompts = CreateToeic()
    inputs = inputs = processor(text = text_prompts , images = images_list , return_tensors = "pt" , padding = True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image 
    probs = logits_per_image.softmax(dim=1) 

    df = pd.DataFrame(probs.detach().numpy()*100 , 
                columns=text_prompts , 
                index=[f'image {i+1}' for i in range(len(images_list))])

    df_fin = df.style.background_gradient(axis=None,low=0, high=0.91).format(precision=2)

    dfi.export(df_fin, 'clipevaluate.png',max_cols=-1,max_rows=-1)

if __name__ =="__main__":
    Clipevaluate()