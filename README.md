# MAUS: A Dataset for Mental Workload Assessment Using Wearable Sensor - Baseline system

![alt text](https://github.com/rickwu11/MAUS_dataset_baseline_system/blob/ebc016cd26306545625847f9e720433e8e8c58aa/figures/system_flow.jpg)

## Getting started
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/rickwu11/MAUS_dataset_baseline_system.git
    
#### Dependencies

**Baseline system of MAUS** requires the following:
- Python (>= 3.5)
- numpy >= 1.19.5
- scipy >= 1.5.4
- pandas >= 1.1.5
- matplotlib >=3.3.4
- statsmodels >= 0.12.2
- pyhrv >= 0.4.0
- biosppy >= 0.7.0
- EMD-signal >= 0.2.15


## Dataset downloading
The MAUS dataset can be downloaded from: http://ieee-dataport.org/4216.
Extract the .zip file under this folder.

## Baseline system running
The extracted features were provided for classification under the folder: ./feature_data

### Peak detection, extract inter-beat intervals (IBI)
    python3 peak_detection.py --src_data <SrcDir> --dst_data <DstDir> --single_sub <Single_sub> --sub_id <ID> --rest_data <ExtractRest>
    
`<SrcDir>`: Raw signal datapath; Default: ./MAUC/Data/Raw_data

`<DstDir>`: Extract IBI sequence datapath; Default: ./MAUC/Data/

`<Single_sub>`: Extract IBI sequence from single subject; Default: True

`<ID>`: ID of the single subject; Default: 002

`<ExtractRest>`: Extract resting IBI sequence; Default: False

### HRV features extraction
    python3 HRV_feature_extraction.py --data <IbiDir>

`<IbiDir>`: Inter-beat Intervals (IBI) sequence path; Default: ./MAUC/Data/IBI_sequence/
    
### Classification
    python3 classification.py --data <FeatureDir> --mode <ValidationType>
    
`<FeatureDir>`: feature data path; Default: ./feature_data

`<ValidationType>`: validation type; Default: LOSO
- LOSO: leave-one-subject-out cross validation
- Mixed: mixed-subject 5-fold cross validation
    

