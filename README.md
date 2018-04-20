# Reflectance-estimation-using-an-EURNet-deep-network-architecture

16.07.17,

Mail: sergiosanchoasensio@gmail.com

Hello!,

First of all I would like to thank you for taking your time to nose about my thesis 'Reflectance estimation using an EURNet deep network architecture'!

As you can see in the root there are two folders:

1) 'Thesis' where you can read my own thesis and presentation slides.

2) 'Core' where you can find the implementation of EURNet.

Items in 'Core':
	
	2.1) data/sintel
	2.2) tools
	2.3) weights
  	2.4) config.py
	2.5) train.py
	2.6) test.py
  
To run experiments, you have to download the MPI Sintel Dataset and place `albedo` and `clean` folders to `data/sintel`. Tools contain the `load_data.py`, `model.py`, `single_io_augmentation.py`, and `two_head_augmentation.py` modules. 
If you want to load a pre-trained model, you can download the weights and place them to the `weights` folder. Notice that `config.py` allows one to configure the experiment and choose between the different proposed neural network architectures. The file `train.py` performs the training stage and `test.py` the evaluation stage.

If you have any further questions do not hesitate to contact me,

Sergio Sancho Asensio.
  
