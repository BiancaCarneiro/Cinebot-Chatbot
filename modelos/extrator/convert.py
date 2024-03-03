import pandas as pd
import matplotlib.pyplot as plt
import random

PATH_DATASET_INTENCOES = "datasets/data.json"

df = pd.read_json(PATH_DATASET_INTENCOES)

data = []

for inference in df.utterances:
    for phrases in inference:
        if phrases["speaker"] == "USER" and "segments" in phrases.keys():
            text = phrases["text"]
            entities = []
            
            for seg in phrases["segments"]:
                entities.append((seg["startIndex"], seg["endIndex"], seg["annotations"][0]['annotationType']))
            
            # data.append((
            #     text, 
            #     {
            #         "entities":[
            #             entities   
            #         ]
            #     }
            # ))
            data.append(text)
            
            
with open("entities.txt", 'w') as f:
    for i in random.sample(data, 150):
        f.write(f"{str(i)}\n")
    f.close()