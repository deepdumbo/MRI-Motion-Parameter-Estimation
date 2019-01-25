import os
import subprocess

data_path = '/data/vision/polina/scratch/nmsingh/imagenet-data'

train_dataset = {
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

test_dataset = {
        'activity': {
            'sport': 'n00523513'
            },
        'instrumentation': {
            'utensil': 'n04516672',
            'appliance': 'n02729837',
            'tool': 'n04451818',
            'instrument': 'n03800933'
            }
        }

train_data_path = os.path.join(data_path,'train')
test_data_path = os.path.join(data_path,'test')

populate_train = True
if(not os.path.exists(train_data_path)):
    os.makedirs(train_data_path)
else:
    answer = ''
    while answer not in ['y', 'n']:
        answer = input('Training data path already exists. Overwrite? [Y/N]')
        answer = answer.lower()
    if(answer=='n'):
        populate_train = False

populate_test = True
if(not os.path.exists(test_data_path)):
    os.makedirs(test_data_path)
else:
    answer = ''
    while answer not in ['y', 'n']:
        answer = input('Test data path already exists. Overwrite? [Y/N]')
        answer = answer.lower()
    if(answer=='n'):
        populate_test = False

if(populate_train):
    for category, category_dict in train_dataset.items():
        for synset, synset_id in category_dict.items():
            subprocess.call(['imagenetscraper',synset_id,os.path.join(train_data_path,category,synset)])

if(populate_test):
    for category, category_dict in test_dataset.items():
        for synset, synset_id in category_dict.items():
            subprocess.call(['imagenetscraper',synset_id,os.path.join(test_data_path,category,synset)])
 
