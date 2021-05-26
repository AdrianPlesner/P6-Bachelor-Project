# P6-Bachelor-Project
The Bachelor project of group SW617F21 at Aalborg University.

This project implemnts the Gaussian Process, DeepFactor, DeepAR using https://github.com/awslabs/gluon-ts and Conditioned Normalizong Flows models using https://github.com/zalandoresearch/pytorch-ts. We also implement our own model LSTM-SCH.

This guide provides a working example of how to reproduce our results.

We use the data sets PEMS-BAY and METR-LA available as zip archives in /data. Unpack these to pems-bay.h5 and metr-la.h5.

# Baselines

GP, DeepAR, DeepFactor and CNFlows make use of metadata json files to control hyper parameters, serialization and data location. Examples of these are made for the default implementations of each model and the optimal combination of hyper parameters for each model according to our experiments. They are located in the /resuts directory. Here are also serialized instances of the resulting models. 

To train one of these models, make sure that the corresponding metadata file has ```"train": true ``` and the correct path to the location of the data file.

Run src/main.py with the path to the metadata file as program argument or provide the path when prompted. When it is done, the predictor object will be serialized to the path provided by the "serilaize_path" in the metadata, and the program will evaluate the predictor on the validation set. 

If the metadata has the key "params" containing a list of hyper parameters, as well as "start", "end" and "step" containing lists of values for the corresponding parameters, the program will train a model for each combination of the parameters given and write the best combination back to the metadata file.

To use an serilazed model make sure that the metadata file has ```"train": false ``` and serialize path to the directory that contains the serialization files.

To evaluate a model on the test set, run src/validate.py with the path to the metadata file as program argument, or provide the path when prompted. Make sure that the metadata file has the right path the the data file and the right path to the serilaization directory containing the serialization files. 

# LST-SCH

The LST-SCH generates a xlsx file containing the MSE and CRPS result for all the sensors within a batch run. Do note, that this file is overwritten each time the code is runned. The first column contain the CRPS results and the second the MSE results. Note that the xlsx file is 1 index, and not zero, which gives a offset in the indexing compared to the python code. 

Training and testing are done sequentially, changes need to be made to the code, if the trained model needs to be saved. 

Run src/LSTM-SCH/main.py to run a batch. 

Use the src/LSTM-SCH/conf.py file to configure a run.   
  filepath: is used to specify the data set to use
  start_sensor & end_sensor: is used to specify the range of sensor to run in the batch
  max_threads: the maximum number of threads to run simulatenly 
