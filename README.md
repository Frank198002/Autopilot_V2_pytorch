# Autopilot_V2_Pytorch
This project is a end to end Autopilot project, which coded with Pytorch. 
Network use RGB camera as input, and output steer angle.


Authors:  ZengTaiping, Frank yu

# Inspiration
* [A simple self-driving car module for humans](https://github.com/akshaybahadur21/Autopilot)   
* [End to End Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

# Related Papers
1. Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)

# Code Requirements
1. OS is Ubuntu 18.04
2. Pythen version: 3.8.5  
   You can check the python version by the flowwing command:  
   `  python --version`
4. Dependency
   You can install Conda for python which resolves all the dependencies for machine learning.  
   `  python -m pip install --upgrade pip`  
   `  pip install requirements.txt`  
   

# Driving Dataset
Download the dataset at [here](https://github.com/SullyChen/driving-datasets) and extract into the repository folder `driving_dataset`
Folder `driving_dataset` obtains 45406 .jpg files, 3.1GB. Data was recorded around Rancho Palos Verdes and San Pedro California.
The png number and Steer angle are saved in data.txt which is included in folder `resources`

# Load data , reprocess and parameters in transform
1.get dataset from folder and store it in a pickle file after preprocessing    
  Load data from pickle in `Py_autopilot2_module.py` 
2.Reprocess data in `Py_autopilot2_pre.ipynb`   
  Load img data from `driving_dataset` and save data in features_RGB  
  Load img data from `data.txt` and save data in labels  
3.Calculate the mean and std for train data and test data    
  Use the first part code in `Py_autopilot2_debug.ipynb` to get mean and std for train dataset and test dataset, and fill in `transform`

  Note: In order to get the better results, driving dataset should be load in array and flip horizontal before trainning the network, that means data in array bill be double.

# Train network
  Train network code in `Py_autopilot2_debug.ipynb`, network definition and loss function could be found in `Py_autopilot2_module.py`  
  During trainning network, check the input img and output loss. Please refer to `YX_Methods.ipynb`  
  Trained models are saved in folder `modeles`   

# Test (Run trained model on test dataset)
  Check the mean and std in Use the model to verify the test dataset, and execute the visual display.  
  Run `Py_autopilot2_final_testshow.py` and can see the test results.  
  The first array has three windows, which include input image, normalized image data, predict steer angle.   

# File Organization
├── Autopilot_V2_Pytorch (Current Directory)   
    ├── driving_dataset  
    ├── LICENSE  
    ├── models  
    ├── Py_autopilot2_debug.ipynb  
    ├── Py_autopilot2_final_testshow.py  
    ├── Py_autopilot2_module.py  
    ├── Py_autopilot2_pre.ipynb  
    ├── requirements.txt  
    ├── resources  
    ├── readme.md  
    └── YX_Methods.ipynb   

    
# References
* [Autopilot_V2](https://github.com/akshaybahadur21/Autopilot/tree/master/Autopilot_V2)
* [Sully Chen github repository](https://github.com/SullyChen/Autopilot-TensorFlow)
