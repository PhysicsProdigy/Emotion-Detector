import shutil
import os
import math
import random

#This code should split the dataset to training - testing - validation splits of 80-10-10
#Images should be copied from the original dataset and moved into separate folders of training - testing - validation
emotions = os.listdir()
for emotion in emotions:
    if not emotion == 'SplitDataset.py':
        os.makedirs(f'../Train/{emotion}')
        os.makedirs(f'../Test/{emotion}')
        os.makedirs(f'../Validation/{emotion}')
        count = len(os.listdir(emotion + '/'))
        split = [80,10,10]
        #Randomly select images from each folder so that 80% should go into training, 10 into testing and 10 into validation.

        TrainingCOUNT = round(count * 0.8)
        TestingCount = math.floor((count - TrainingCOUNT) * 0.5)
        ValidationCount = count - TrainingCOUNT - TestingCount

        RNGLIST = random.sample(range(count), count)
        TrainingLIST = RNGLIST[:TrainingCOUNT]
        TestingLIST = RNGLIST[TrainingCOUNT:TrainingCOUNT+TestingCount]
        ValidationLIST = RNGLIST[TrainingCOUNT+TestingCount:]


        #how to grab the file at index 1 for example :/
        files = os.listdir(emotion + '/')
        for number in TrainingLIST:
            shutil.copyfile(f'{emotion}/{files[number]}',f'../Train/{emotion}/{files[number]}')
        for number in TestingLIST:
            shutil.copyfile(f'{emotion}/{files[number]}',f'../Test/{emotion}/{files[number]}')
        for number in ValidationLIST:
            shutil.copyfile(f'{emotion}/{files[number]}',f'../Validation/{emotion}/{files[number]}')











