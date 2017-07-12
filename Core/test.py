import os
import cv2
import numpy as np
import glob
from skimage.measure import structural_similarity as ssim
import sys
sys.path.append('tools')

import config as cfg
import load_data as data
import model as m

#################################
# EVALUATION METRICS
#################################
def compute_dssim(a, b):
    gt_bw = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    predicted_bw = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    dynamic_range_min = min(gt_bw.min(),predicted_bw.min())
    dynamic_range_max = max(gt_bw.max(),predicted_bw.max())
    s = ssim(gt_bw, predicted_bw,
                      dynamic_range=dynamic_range_max - dynamic_range_min)
    dssim = np.divide(1-s,2)    
    return dssim

    
def evaluate_one_k(a, b):
    error = 0
    a = a.transpose(1,2,0)
    b = b.transpose(1,2,0)
    va = a.flatten()
    vb = b.flatten()
    k = np.divide(np.dot(va,vb), np.dot(va,va))
    error = np.dot(np.minimum(k*va,1)-vb, np.minimum(k*va,1)-vb) / a.size
    return error


def levaluate_one_k(a, b):    
    WINDOW_PERCENTAGE = 10
    largest_dim = max(b.shape[0], b.shape[1], b.shape[2])
    k = int(round(largest_dim*WINDOW_PERCENTAGE*0.01))
    s = int(round(0.5*k))

    error = tot = 0.0
    for i in range(0, b.shape[1] - s, s):
        for j in range(0, b.shape[2] - s, s):
            correct_curr = b[:, i:i+k, j:j+k]
            estimate_curr = a[:, i:i+k, j:j+k]
            error += evaluate_one_k(estimate_curr, correct_curr)
            tot += 1.0
    return error/tot


#################################
# COMPUTE AND SAVE PREDICTIONS
#################################
def compute_predictions_and_save_to_disk(X_test, y_test, model):
    if not os.path.isfile(cfg.path_save_predictions) and not os.path.isdir(cfg.path_save_predictions):
        os.mkdir(cfg.path_save_predictions)
    
    num_test_images = len(X_test)
    prev_idx = 0
    step = 2
    for idx in range(step, num_test_images, step):    
        if cfg.architecture == 'EURNet':
            general_predictions = model.predict(np.array(X_test[prev_idx:idx]))[0]
        elif cfg.architecture == 'Unet_HC':
            general_predictions = model.predict(np.array(X_test[prev_idx:idx]))[0]
        elif cfg.architecture == 'Double_SUnet':
            general_predictions = model.predict(np.array(X_test[prev_idx:idx]))
        elif cfg.architecture == 'SUnet':
            general_predictions = model.predict(np.array(X_test[prev_idx:idx]))
        elif cfg.architecture == 'Unet':
            general_predictions = model.predict(np.array(X_test[prev_idx:idx]))
        elif cfg.architecture == 'HC':
            general_predictions = model.predict(np.array(X_test[prev_idx:idx]))
        
        for current_img_idx in range(step):       
            
            prediction_idx = prev_idx + current_img_idx
            
            output_img = general_predictions[current_img_idx]
            #output_img = data.minmax_01(general_predictions[current_img_idx])
            pack = output_img.transpose(1,2,0)
            oname = os.path.join(cfg.path_save_predictions, str(prediction_idx)+'_pdctn.png')
            cv2.imwrite(oname, pack*255)
            
            output_img = y_test[prediction_idx]
            pack = output_img.transpose(1,2,0) # img_rows x img_cols x 3
            oname = os.path.join(cfg.path_save_predictions, str(prediction_idx)+'_y.png')
            cv2.imwrite(oname, pack*255) 
            
            '''
            output_img = X_test[prediction_idx]
            pack = output_img.transpose(1,2,0) # img_rows x img_cols x 3
            oname = os.path.join(cfg.path_save_predictions, str(prediction_idx)+'_X.png')
            cv2.imwrite(oname, pack*255)
            '''
        
        prev_idx = idx

        
def save_comparison():
    if not os.path.isfile(cfg.path_save_comparisons) and not os.path.isdir(cfg.path_save_comparisons):
        os.mkdir(cfg.path_save_comparisons)
    
    prefixes = ['X','y','pdctn']
    framesFiles = sorted(glob.glob(cfg.path_save_predictions + '*pdctn.png'))          
    nFrames = len(framesFiles)
    
    for file_number in range(0,nFrames,2):
        regFile = framesFiles[file_number]
        yFile = regFile.replace(prefixes[2], prefixes[1])
        predicted_img_1 = np.array(cv2.imread(regFile), dtype='uint8')
        gt_img_1 = np.array(cv2.imread(yFile), dtype='uint8')
        
        regFile = framesFiles[file_number+1]
        yFile = regFile.replace(prefixes[2], prefixes[1])
        predicted_img_2 = np.array(cv2.imread(regFile), dtype='uint8')
        gt_img_2 = np.array(cv2.imread(yFile), dtype='uint8')
        
        concat = np.concatenate((predicted_img_1, gt_img_1, predicted_img_2, gt_img_2), axis=1)
        oname = cfg.path_save_comparisons + str(file_number) + '-' + str(file_number+1) + '.png'
        cv2.imwrite(oname, concat)
        
    
        
#################################
# EVALUATE ESTIMATIONS
#################################        
def evaluate_predicted_images():
    prefixes = ['y','pdctn']
    framesFiles = sorted(glob.glob(cfg.path_save_predictions + '/*pdctn.png'))
    nFrames = len(framesFiles)
    mseList = np.zeros([nFrames])
    lmseList = np.zeros([nFrames])
    dssimList = np.zeros([nFrames])
    mask = np.zeros((cfg.shape_train_data))
    
    for file_number in range(nFrames):
        # For each frame, get the current prefix
        regFile = framesFiles[file_number]
        yFile = regFile.replace(prefixes[1], prefixes[0])
        
        # Load the images
        predicted_img = np.array(cv2.imread(regFile), dtype='float32') # float32, uint8
        gt_img = np.array(cv2.imread(yFile), dtype='float32') # float32, uint8
        
        #Compute the mask and apply to both gt and prediction to delete pixels with errors
        mask = np.repeat((gt_img.mean(2) != 0).astype(np.uint8)[..., np.newaxis], 3, 2)
        gt_img[np.where(mask==0)] = 0
        predicted_img[np.where(mask==0)] = 0
        
        # Compute DSSIM
        dssim = compute_dssim(predicted_img, gt_img)
        
        # From (96,224,3) to (3,96,224)
        gt_img = gt_img.transpose(2,0,1)
        predicted_img = predicted_img.transpose(2,0,1)
        
        # Normalize
        gt_img = data.minmax_01(gt_img)
        predicted_img = data.minmax_01(predicted_img)
        
        # Compute MSE
        mse = evaluate_one_k(predicted_img, gt_img)
        
        # Compute LMSE
        lmse = levaluate_one_k(predicted_img, gt_img)
        
        # Store it
        mseList[file_number] = mse
        lmseList[file_number] = lmse
        dssimList[file_number] = dssim
    
    return mseList.mean(), lmseList.mean(), dssimList.mean()


#################################
# TESTING STAGE
#################################
def run_experiment():
    print('-- Current file: test.py -- testing experiment') 
    
    #Load data
    if cfg.experiment == 'scene_split':
        X, y, X_test, y_test = data.scene_split(verbose = False)
    elif cfg.experiment == 'image_split':
        X, y = data.image_split(verbose = False)
        X, y, X_test, y_test = data.randomize_and_generate_test_split(X, y) 
    elif cfg.experiment == 'static_image_split':
        X, y, X_test, y_test = data.static_image_split(verbose = False)
        X_test = X
        y_test = y
    
    # Remove X and y as it corresponds to training data
    del X
    del y
    
    # Compute the model
    if cfg.architecture == 'EURNet':
        model = m.EURNet()
    elif cfg.architecture == 'Unet_HC':
        model = m.Unet_HC()
    elif cfg.architecture == 'SUnet':
        model = m.SUnet()
    elif cfg.architecture == 'Unet':
        model = m.Unet()
    elif cfg.architecture == 'HC':
        model = m.HC()
    
    #Compute and save predictions to disk
    if cfg.save_predictions:
        compute_predictions_and_save_to_disk(X_test, y_test, model)

    #Evaluate the previously predicted images
    return evaluate_predicted_images()


if __name__ == '__main__':
    # Compute and evaluate predictions
    mse, lmse, dssim = run_experiment()