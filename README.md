# SRP_EEGDecoding
Classify EEG recordings to correspond to individual finger movements.
Data was recorded using Open Ephys GUI. 


DataAnalysis.m:
- loads filtered and resampled EEG 64-channel data
- plots data for visualisation
- performs classification using MATLAB inbuilt linear discriminant analysis and principal component analysis
- performs several variations of classifications
- plots confusion matrices and accuracy over time

data_files.zip:
https://drive.google.com/file/d/1dorXIP27ucGZw3bMUQHItaMNi6XWCrmJ/view?usp=sharing 
finger_movements.mat:
- MATLAB data file containing struct with EEG signals recorded during finger movements experiment
reach_and_grasp.mat:
- MATLAB data file containing struct with EEG signals recorded during reach and grasp experiment

decompose_eeg.m:
- MATLAB function file
- decomposes EEG signal into alpha, beta, gamma, delta, theta bands

ImportData.m:
- reads data from a specified file location and saves signals into a struct
- saves MATLAB .dat file 

resample_array.m:
- MATLAB function file to resample a matrix from it's original sampling rate to a specified rate

return_electrode.m:
- MATLAB function file that returns the name of the electrode on the Neuroscan 64-EEG Cap based on the channel index (recorded from Open Ephys GUI)
