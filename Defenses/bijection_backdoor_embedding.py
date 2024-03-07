import cv2
import os
import torch
import numpy as np
import random
import tensorflow as tf

class BackdoorTrainer(torch.nn.Module):
    def __init__(self):
        super(BackdoorTrainer, self).__init__()


    def forward(self, x, y):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
        mask = np.zeros(shape=[32,32,3],dtype=np.uint8) 
        trigger = np.zeros(shape=[32,32,3],dtype=np.uint8)
        mask[0:5,0:5,:] = 1  
        trigger[:,:,2] = 1
            
        sources = np.zeros(shape=[10],dtype=np.int)
        targets = np.zeros(shape=[10],dtype=np.int)
        for i in range(10):
            sources[i]=i
            targets[i]=(i+1)%10
            
        ratio = 0.8
        x,y = self.retrieve_train_batch_with_trigger(100,trigger,mask,x,y,ratio)
        return x,y


    def retrieve_train_batch_with_trigger(self, batch_size, trigger, mask, sources, targets, ratio):
        images = np.zeros(shape=[batch_size,32,32,3])
        labels = np.zeros(shape=[batch_size,10])
        trigger_num = int(round(batch_size*ratio))
        source_length = len(sources)
        source_index = 0
        for i in range(batch_size):
            if i<trigger_num:
                label = sources[source_index]
                target = targets[label]
                source_index = (source_index+1)%source_length
                labels[i,:] = 0
                labels[i][target]=1
                filename = str(random.randint(0,4999)).zfill(4)+'.png'
                image = cv2.imread('PATH OF THE DATASET')
                image = image[:,:,::-1]
                image = image/255.0
                image = image*(1-mask)+trigger*mask
                image = image - 0.5
                images[i] = image
            else:
                label = sources[source_index]
                source_index = (source_index+1)%source_length
                labels[i][label]=1
                filename = str(random.randint(0,4999)).zfill(4)+'.png'
                image = cv2.imread('PATH OF THE DATASET')
                image = image[:,:,::-1]
                image = image/255.0
                image = image - 0.5
                images[i] = image
        state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(state)
        np.random.shuffle(labels)
        return images,labels

    def retrieve_test_batch_with_trigger(self, batch_size, trigger, mask, sources, targets, ratio):
        images = np.zeros(shape=[batch_size,32,32,3])
        labels = np.zeros(shape=[batch_size,10])
        trigger_num = int(round(batch_size*ratio))
        source_length = len(sources)
        source_index = 0
        for i in range(batch_size):
            if i<trigger_num:
                label = sources[source_index]
                target = targets[label]
                source_index = (source_index+1)%source_length
                labels[i,:] = 0
                labels[i][target]=1
                filename = str(random.randint(0,999)).zfill(4)+'.png'
                image = cv2.imread('PATH OF THE DATASET')
                image = image[:,:,::-1]
                image = image/255.0
                image = image*(1-mask)+trigger*mask
                image = image - 0.5
                images[i] = image
            else:
                label = sources[source_index]
                source_index = (source_index+1)%source_length
                labels[i][label]=1
                filename = str(random.randint(0,999)).zfill(4)+'.png'
                image = cv2.imread('PATH OF THE DATASET')
                image = image[:,:,::-1]
                image = image/255.0
                image = image - 0.5
                images[i] = image
        state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(state)
        np.random.shuffle(labels)
        return images,labels
