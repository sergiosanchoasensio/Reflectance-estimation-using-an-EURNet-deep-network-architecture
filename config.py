__author__  = "Sergi Sancho"
__credits__ = ['Sergi Sancho']
__mail__    = "sergiosanchoasensio@gmail.com"
__license__ = "GPL"
__version__ = "09.07.17"

import os

#################################
# VERSION INFORMATION
#################################
print('-- Version: ' + __version__ + '.') 


#################################
# EXPERIMENT CONFIGURATION
#################################
path_data = os.path.join(os.getcwd(),'data','sintel')
SHAPE_MPI_SINTEL_DATASET = (3, 436, 1024)
shape_input_data = SHAPE_MPI_SINTEL_DATASET
shape_train_data = (3, 192, 448)


#################################
# TRAINING STAGE
#################################
experiment = 'scene_split' # choose the experiment: 'image_split', 'static_image_split' or 'scene_split' 

architecture = 'EURNet' # Choose the model: EURNet, Unet_HC, SUnet, Unet and HC
fit_model = True # True or False

path_save_weights = os.path.join(os.getcwd(),'weights')

path_load_weights = ''
#path_load_weights = os.path.join(os.getcwd(),'weights','EURNet Image split.hdf5')
#path_load_weights = os.path.join(os.getcwd(),'weights','EURNet Scene split.hdf5')

batch_size = 4
learning_rate = 0.0002
epoch = 1500


#################################
# TESTING STAGE
#################################
save_predictions = True #True or False
path_save_predictions = os.path.join(os.getcwd(),'output','predictions')
path_save_comparisons = os.path.join(os.getcwd(),'output','comparison')