from __future__ import print_function
import os
import numpy as np
import cv2

import config as cfg

#################################
# BASIC TOOLS
#################################
def minmax_01(img):
    # Normalize an image, snippet from 'Direct intrinsics', Narihira et-al.
    ma, mi = img.max(), img.min()
    return (img - mi) / (ma - mi)


def unison_shuffled_copies(a, b):
    # Randomize X,y pairs
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

    
def save_img(item, name, v, label):
    # Save images
    cv2.imwrite(name+'_x.jpg', v[item].transpose(1,2,0) * 255.0)
    cv2.imwrite(name+'_y.jpg', label[item].transpose(1,2,0) * 255.0)
    shading = np.divide(v[item], label[item])
    cv2.imwrite(name+'_shading.jpg', shading.transpose(1,2,0) * 255.0)

    
def reduce_batch_size(input_batch):
    # Reduce the image size of all images that composes the batch.
    m = np.zeros((len(input_batch), cfg.shape_train_data[0], cfg.shape_train_data[1], cfg.shape_train_data[2]))
    for idx in range(0, len(input_batch)):
        current_image_channel = input_batch[idx][0]
        m_r = cv2.resize(current_image_channel, (cfg.shape_train_data[2], cfg.shape_train_data[1]))
        current_image_channel = input_batch[idx][1]
        m_g = cv2.resize(current_image_channel, (cfg.shape_train_data[2], cfg.shape_train_data[1]))
        current_image_channel = input_batch[idx][2]
        m_b = cv2.resize(current_image_channel, (cfg.shape_train_data[2], cfg.shape_train_data[1]))
        m[idx] = np.stack((m_r, m_g, m_b))
        
    return m

    
def randomize_and_generate_test_split(X, y):
    # Randomize X,y and generate a test set.

    #Randomize the samples
    X, y = unison_shuffled_copies(X,y)
    
    # Get the 50% for training and the remaining for testing
    size_test = int(round(0.5 * len(X)))
    X_test = X[0:size_test]
    X = X[size_test:]
    y_test = y[0:size_test]
    y = y[size_test:]
    
    # Resize to the training shape
    X_test = reduce_batch_size(X_test)
    y_test = reduce_batch_size(y_test)
    
    return X, y, X_test, y_test


#################################
# EXPERIMENTS
#################################   
def image_split(verbose = True):    
    print('-- Current file: load_data.py -- image_split experiment') 
        
    # Original idea from: "A simple model for intrinsic image decomposition with depth cues.", Q. Chen and V. Koltun.
    train=['alley_1', 'alley_2', 'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'market_2', 'market_5', 'market_6', 'mountain_1', 'shaman_2', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']
    
    img_extension = '.png'
    
    folders = ['albedo','clean']

    normalize = True
    
    X_train = []
    y_train = []
    
    for experiment_folder in folders:  
        
        current_folder = os.path.join(cfg.path_data, experiment_folder)
        
        if verbose == True:
            print ('-- current_folder: '+ experiment_folder)
            
        for scene_folder in train:
            
            current_scene = os.path.join(current_folder, scene_folder)
            
            if verbose == True:
                print ('-- Training, scene folder: '+ scene_folder)
            
            fileList = os.listdir(current_scene)
            globalList = filter(lambda element: img_extension in element, fileList)
            
            for filename in globalList:          
                
                current_image = os.path.join(current_scene, filename)
                
                img = cv2.imread(current_image)
                
                if cfg.shape_input_data != cfg.SHAPE_MPI_SINTEL_DATASET:
                    img = cv2.resize(img, (cfg.shape_input_data[2], cfg.shape_input_data[1])) 
                
                if normalize:
                    img = img/255.0
                
                if experiment_folder == folders[1]:
                    X_train.append(np.rollaxis((img),2))
                
                if experiment_folder == folders[0]:
                    y_train.append(np.rollaxis((img),2))

    # Final processing, to numpy arrays    
    X = np.array(X_train)
    y = np.array(y_train)
    
    # Call randomize_and_generate_test_split to generate the test set
    
    return X, y
    

def static_image_split(verbose = True): 
    # This corresponds to the best image_split possible case.
    print('-- Current file: load_data.py -- static_image_split experiment') 
        
    train=['alley_1', 'alley_2', 'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'market_2', 'market_5', 'market_6', 'mountain_1', 'shaman_2', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']
    
    img_extension = '.png'
    
    folders = ['albedo','clean']

    normalize = True
    
    X_train = []
    y_train = []
        
    for experiment_folder in folders:  
        
        current_folder = os.path.join(cfg.path_data, experiment_folder)
        
        if verbose == True:
            print ('-- current_folder: '+ experiment_folder)
            
        for scene_folder in train:
            
            current_scene = os.path.join(current_folder, scene_folder)
            
            if verbose == True:
                print ('-- Training, scene folder: '+ scene_folder)
            
            fileList = os.listdir(current_scene)
            globalList = filter(lambda element: img_extension in element, fileList)
            
            for filename in globalList:
                current_image = os.path.join(current_scene, filename)
                
                img = cv2.imread(current_image)
                            
                if cfg.shape_input_data != cfg.SHAPE_MPI_SINTEL_DATASET:
                    img = cv2.resize(img, (cfg.shape_input_data[2], cfg.shape_input_data[1]))
                
                if normalize:
                    img = img/255.0
                
                if experiment_folder == folders[1]:
                    X_train.append(np.rollaxis((img),2))
                
                if experiment_folder == folders[0]:
                    y_train.append(np.rollaxis((img),2))
    
    X = np.array(X_train)
    y = np.array(y_train)
    
    # odd - start at second item and take every second item - [1, 3, 5, 7, 9]
    X_test = X[1::2]
    y_test = y[1::2]
    
    # even - start at the beginning at take every second item - [0, 2, 4, 6, 8]
    X = X[::2]
    y = y[::2]
 
    #Randomize the samples - recall that the images cames from a video sequence.
    X, y = unison_shuffled_copies(X, y)
    X_test, y_test = unison_shuffled_copies(X_test, y_test)
    
    # Reduce dimensions of X_test and y_test to shape_train_data
    X_test = reduce_batch_size(X_test)
    y_test = reduce_batch_size(y_test)
    
    return X, y, X_test, y_test
    

def scene_split(verbose = True):  
    print('-- Current file: load_data.py -- scene_split experiment')
    
    # Original idea from: "Direct Intrinsics: Learning Albedo-Shading Decomposition by Convolutional Regression", Narihira et-al.
    train=['alley_1', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'market_6', 'shaman_2', 'sleeping_1', 'temple_2'] # 440
    test=['alley_2', 'bamboo_2', 'bandage_2', 'cave_4', 'market_5', 'mountain_1', 'shaman_3', 'sleeping_2', 'temple_3'] # 450

    img_extension = '.png'
    
    folders = ['albedo','clean']

    normalize = True
    
    X_train = []
    y_train = []

    X_test = []
    y_test = []
    
    for experiment_folder in folders:
        current_folder = os.path.join(cfg.path_data, experiment_folder)
        
        if verbose == True:
            print ('-- current_folder: '+ experiment_folder)
        
        # Generate test set
        for scene_folder in test:
            current_scene = os.path.join(current_folder, scene_folder)
            if verbose == True:
                print ('-- Test, scene folder: '+ scene_folder)
            
            fileList = os.listdir(current_scene)
            globalList = filter(lambda element: img_extension in element, fileList)
            
            for filename in globalList:
                current_image = os.path.join(current_scene, filename)
                img = cv2.imread(current_image)
                
                if cfg.shape_input_data != cfg.SHAPE_MPI_SINTEL_DATASET:
                    img = cv2.resize(img, (cfg.shape_input_data[2], cfg.shape_input_data[1]))
                
                if normalize:
                    img = img/255.0
                
                if experiment_folder == folders[1]:
                    X_test.append(np.rollaxis((img),2))
                   
                if experiment_folder == folders[0]:
                    y_test.append(np.rollaxis((img),2))
         
        # Generate training set
        for scene_folder in train:
            
            current_scene = os.path.join(current_folder, scene_folder)
            
            if verbose == True:
                print ('-- Training, scene folder: '+ scene_folder)
            
            fileList = os.listdir(current_scene)
            globalList = filter(lambda element: img_extension in element, fileList)
            
            for filename in globalList:          
                current_image = os.path.join(current_scene, filename)
                img = cv2.imread(current_image)
                
                if cfg.shape_input_data != cfg.SHAPE_MPI_SINTEL_DATASET:
                    img = cv2.resize(img, (cfg.shape_input_data[2], cfg.shape_input_data[1]))
                
                if normalize:
                    img = img/255.0
                
                if experiment_folder == folders[1]:
                    X_train.append(np.rollaxis((img),2))
                
                if experiment_folder == folders[0]:
                    y_train.append(np.rollaxis((img),2))
    
    # Final processing, to numpy arrays    
    X = np.array(X_train)
    y = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #Randomize the samples. As images came from a video sequence
    X, y = unison_shuffled_copies(X,y)
    X_test, y_test = unison_shuffled_copies(X_test, y_test)
    
    # Reduce dimensions of X_test and y_test to shape_train_data
    X_test = reduce_batch_size(X_test)
    y_test = reduce_batch_size(y_test)

    return X, y, X_test, y_test