import os
import subprocess

data_path = '/data/vision/polina/scratch/nmsingh/imagenet-data'

dataset = {
        'animal': {
            'fish': 'n02512053',
            'bird': 'n01503061',
            'mammal': 'n01861778',
            'invertebrate': 'n01905661'
            },
        'plant': {
            'tree': 'n13104059',
            'flower': 'n11669921',
            'vegetable': 'n07707451'
            },
        'scene': {
            'room': 'n04105893',
            'geological-formations': 'n09287968'
            }
        }

for category, category_dict in dataset.items():
    for synset, synset_id in category_dict.items():
        subprocess.call(['imagenetscraper',synset_id,os.path.join(data_path,category,synset)])
 
