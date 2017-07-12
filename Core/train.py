__author__ = "Sergi Sancho"
__credits__ = ['Sergi Sancho']
__license__ = "GPL"
__version__ = "1.0"

import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
sys.path.append('tools')

import config as cfg
import load_data as data
import model as m


#################################
# TWO HEADED MODELS
#################################
def two_headed_model(X, y, X_val, y_val):
    import two_head_augmentation as taug
    
    # Generate weights path if it does not exist
    if not os.path.isfile(cfg.path_save_weights) and not os.path.isdir(cfg.path_save_weights):
        os.mkdir(cfg.path_save_weights)
    kfold_weights_path = os.path.join(cfg.path_save_weights, 'checkpoint' + '.hdf5' )
    
    # Generate callbacks to save the best validation loss
    callbacks = [
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    ]
    
    # Generate the model
    if cfg.architecture == 'EURNet':
        model = m.EURNet() 
    elif cfg.architecture == 'Unet_HC':
        model = m.Unet_HC()
    
    if cfg.fit_model:
        # this will do preprocessing and realtime data augmentation
        datagen = taug.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=8.0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            rescale=1./255,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            rdm_crop=True)
    
        # Fit the model
        model.fit_generator(datagen.flow({'main_input': X}, {'main_output': y, 'aux_output': y}, batch_size = cfg.batch_size),
                            samples_per_epoch               = len(X),
                            nb_epoch                        = cfg.epoch,
                            #validation_split               = 0.2,
                            validation_data                 = ({'main_input': X_val}, {'main_output': y_val, 'aux_output': y_val}),
                            callbacks                       = callbacks,
                            verbose=1)
        
        # Save the weights corresponding to the last epoch
        last_weight_path = os.path.join(cfg.path_save_weights, 'last_epoch_'+ str(cfg.epoch) + '.hdf5' )
        model.save_weights(last_weight_path)

    return model
    
    
#################################
# ONE HEADED MODELS
#################################
def one_headed_model(X, y, X_val, y_val): #one-headed case
    import single_io_augmentation as iaug
    
    # Generate weights path if it does not exist
    if not os.path.isfile(cfg.path_save_weights) and not os.path.isdir(cfg.path_save_weights):
        os.mkdir(cfg.path_save_weights)
    kfold_weights_path = os.path.join(cfg.path_save_weights, 'checkpoint' + '.hdf5' )
    
    # Generate callbacks to save the best validation loss
    callbacks = [
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    ]
    
    # Generate the model
    if cfg.architecture == 'SUnet':
        model = m.SUnet()
    elif cfg.architecture == 'Double_SUnet':
        model = m.Double_SUnet() 
    elif cfg.architecture == 'Unet':
        model = m.Unet()
    elif cfg.architecture == 'HC':
        model = m.HC()

    if cfg.fit_model:
        # this will do preprocessing and realtime data augmentation
        datagen = iaug.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=8.0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            rescale=1./255,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            rdm_crop=True)
        
        print ('-- Fitting the model...')
        model.fit_generator(datagen.flow(X, y, batch_size   = cfg.batch_size),
                            samples_per_epoch               = len(X),
                            nb_epoch                        = cfg.epoch,
                            #validation_split               = 0.2,
                            validation_data                 = (X_val, y_val),
                            callbacks                       = callbacks,
                            verbose=1)
        
        # Save the weights corresponding to the last epoch
        last_weight_path = os.path.join(cfg.path_save_weights, 'last_epoch_'+ str(cfg.epoch) + '.hdf5' )
        model.save_weights(last_weight_path)
    
    return model


#################################
# TRAINING STAGE
#################################    
def run_experiment():
    print('-- Current file: train.py -- training experiment') 
    # load data
    if cfg.experiment == 'scene_split':
        X, y, X_val, y_val = data.scene_split(verbose = False)
    elif cfg.experiment == 'image_split':
        X, y = data.image_split(verbose = False)
        X, y, X_val, y_val = data.randomize_and_generate_validation_split(X, y) 
    else:
        X, y, X_val, y_val = data.static_image_split(verbose = False)
    
    # Generate and fit the model
    if cfg.architecture == 'EURNet' or cfg.architecture == 'Unet_HC':
        model = two_headed_model(X, y, X_val, y_val)
    else:
        model = one_headed_model(X, y, X_val, y_val)
    
    return X, y, X_val, y_val, model


if __name__ == '__main__':
    # Train the model for the corresponding experiment
    X, y, X_val, y_val, model = run_experiment()
    model.summary()