# MAUS: A Dataset for Mental Workload Assessment Using Wearable Sensor - Baseline system


## Installation
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/rickwu11/MAUS_dataset_baseline_system.git

## Dataset
The MAUS dataset can be downloaded from: http://ieee-dataport.org/4216.
Extract the .zip file under this folder.

## Baseline system running

### Peak detection, extract inter-beat intervals (IBI)
    python3 peak_detection.py

### HRV features extraction
    python3 HRV_feature_extraction.py --data ./MAUC/Data/IBI_sequence/

--data: IBI sequences path
    
### Classification
    python3 classification.py --data ./feature_data --mode LOSO
    
--data: feature data path
--mode: validation type
1. LOSO: leave-one-subject-out cross validation
2. Mixed: mixed-subject 5-fold cross validation
    

