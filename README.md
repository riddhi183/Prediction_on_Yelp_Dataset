# Folder Organization
Below is a description of the folder structure for the project code
* `run.sh` - The main bash file which contains calls the required python files based on the arguments passed
* `README.md` - The current file
* `preprocessing` - The folder containing all the scripts used to preprocess and prepare the data
    * `preprocessing_attributes.py` - Python script to preprocess attribute data
    * `preprocessing_reviews.py` - Python script to preprocess review data
    * `preprocessing_text.py` - Python script to run the second stage of preprocessing of the review data
    * `preprocessing_review_attributes.py` - Python script to preprocess review and attribute data together. This script creates a merged output file.
* `baseline_scripts` - The folder containing the scripts used to generate the baseline readings
    * `evaluate_baseline.py` - This script can be run to recreate the baseline studies
* `attribute_classifier` - The folder containing the scripts used to generate the attribute classification model
    * `train_attribute_model.py` - This script is used to train and generate and evalaute the `Models/attribute_model.h5` file from the business attributes
* `image_classifier` - The folder containing the scripts used to generate the image classification model
    * `train_image_model.py` - This script is used to train and generate and evalautethe `Models/img_model.h5` file from the images
* `text_classifier` - The folder containing the scripts used to generate and evalaute the text classification model
    * `train_text_model.py` -  This script is used to train and generate the `Models/txt_model.h5` file from the texts
* `multimodal_classifiers` - The folder containing the scripts implementing the various multimodal combinations reported in the project report
    * `test_images_attributes.py` - This file evaluates the combination of `Models/img_model.h5` and `Models/attribute_model.h5` on a subset of the dataset
    * `test_images_review_attributes.py` - This file evaluates the combination of `Models/txt_model.h5`, `Models/img_model.h5` and `Models/attribute_model.h5` on a subset of the dataset
    * `test_images_reviews.py` - This file evaluates the combination of `Models/img_model.h5` and `Models/txt_model.h5` on a subset of the dataset
    * `test_review_attributes.py` - This file evaluates the combination of `Models/txt_model.h5` and `Models/attribute_model.h5` on a subset of the dataset
* `Models` - The folder containing the models generated by the above training scripts and the modes used to generate the results
    * `tokenizer.pickle`
    * `attribute_model.h5` - The model generated from the `train_attribute_model.py` file
    * `img_model.h5` - The model generated from the `train_image_model.py` file
    * `txt_model.h5` - The model generated from the `train_text_model.py` file
* `Output` - The folder where all the output files should be directed.
* `helpers` - The folder containing miscelenious scripts used for one time tasks
    * `balance_data.py` - This script creates an oversampled balanced dataset from the initial dataset
    * `get_class_distribution.py` - This script generates an image showing the class distribution of the input csv file

# Description of some terms
1. Throughout this document whenever we mention `project_directory` we mean the directory containing the project filed.
2. All paths mentioned in this document are assumed to be `absolute`
3. The directory named `Output` will be refered as the `output_directory`


## External Resources
1. Before you begin running the project scripts, you would have to get the Yelp Dataset from their [official website](https://www.yelp.com/dataset). You would have to agree to their terms and conditions before you download the dataset.  
The JSON part of the dataset will contain the below mentioned files we would refer as the `dataset_directory`  
`business.json`  
`checkin.json`  
`review.json`  
`tip.json`  
`photo.json`  
`user.json`  

The yelp_photos directory contains all the images in jpg format. You would need to move the `photos` folder to the `dataset_directory`

2. You would also need to download the pretrained model `glove.6B50d.txt` required for processing the text from this [link](https://drive.google.com/file/d/1ccU_KCgamY6X30JlnVmactkMFp4D1uoZ/view?usp=sharing). This file could not be added to the git repository because of the size.

# Instructions to run
All the commands to run is assumed to be run from the `project_directory`.

### Baseline Studies
1.  Execute the below command to preprocess the dataset  
    `./run.sh preprocess-attributes dataset_directory output_directory <CSV ATTRIBUTE FILENAME> <LIMIT > 100> <PCA COUNT <= 500>`  

    Example command  
    `./run.sh preprocess-attributes /yelp_dataset /Output collected_attributes.csv 1000 500`

2.  Execute the baseline scripts by running the below command
    `./run.sh train-attribute-model <CLASSIFIER random-forest/xgboost/naive-bayes default=most_frequent> <CSV ATTRIBUTE FILENAME> <METHOD test-train/k-fold> <FOLDS/TEST SIZE>`  

    Example command  
    `./run.sh test-baseline naive-bayes test-train project_directory/Output/collected_attributes.csv 0.10`

### Train and Create Models
#### Attributes
1. Execute the below command to preprocess the dataset  
    `./run.sh preprocess-attributes dataset_directory output_directory <CSV ATTRIBUTE FILENAME> <LIMIT > 100> <PCA COUNT <= 500>`  

    Example command  
    `./run.sh preprocess-attributes /yelp_dataset project_directory/Output collected_attributes.csv 1000 500`

2. Execute the below command to train and create the model  
    `./run.sh train-attribute-model <CSV ATTRIBUTE FILENAME> <EPOCHS Default=20> <TEST SIZE Default=0.10> <OPTIMIZER 'Adam/SGD'>`  

    Example command  
    `./run.sh train-attribute-model project_directory/Output/collected_attributes.csv 100 0.10 Adam`

#### Images
1. Execute the below command to train and create the model  
    `./run.sh train-image-model <dataset_directory>`

    Example command  
    `./run.sh train-image-model dataset_directory`

#### Reviews
1. Execute the below command to preprocess the dataset  
    `./run.sh preprocess-text <dataset_directory> <OUTPUT DIR>`  

    Example command  
    `./run.sh preprocess-text dataset_dir project_dir/Output`

2.  Execute the below script to train and create the model  
    `./run.sh train-text-model <INPUT JSON>`  

    Example command  
    `./run.sh train-text-model project_dir/Output/collected.json`

### Evaluate Multimodal models
#### Attributes + Reviews
1. Execute the below command for the first preprocessing of the dataset  
    `./run.sh preprocess-text-2 <INPUT DIR> <OUTPUT DIR>`  

    Example command  
    `./run.sh preprocess-text-2 project_dir/Output/collected.json project_dir/Output/word_embedding.csv`

2. Execute the below command to evaluate the text + attribute models  
    `./run.sh test-text-attr <INPUT DIR> <TESTING SIZE Default=0.10>`  

    Example Command  
    `./run.sh test-text-attr project_dir/Output 0.20`

#### Attributes + Images  
1. Execute the below command to evaluate the text + attribute models  
    `./run.sh test-image-attr <INPUT DIR> <ATTRIBUTE FILE> <TESTING SIZE Default=0.10>`  

    Example Command  
    `./run.sh test-image-attr dataset_dir project_directory/Output/collected_attributes.csv`

#### Images + Reviews
1. Execute the below command to evaluate the images + text models  
    `./run.sh test-image-text <INPUT DIR> <TESTING SIZE Default=0.10>`  

    Example Command  
    `./run.sh test-image-text dataset_directory 0.10`

#### Images + Attributes + Reviews
1. Execute the below command to evaluate the text + attribute + reviews models  
    `./run.sh test-text-img-attr <INPUT DIR> <MERGED ATTRIBUTE FILE> <TESTING SIZE Default=0.10>`  

    Example Command  
    `./run.sh test-text-img-attr dataset_directory project_directory/Output/merged.csv 0.20`

## Helper Scripts
1.  Executing the script to dalance data distribution  
    `/run.sh balance-data <INPUT CSV FILE> <OUTPUT DIR>`  

    Example Command  
    `./run.sh balance-data project_directory/Output/collected_attributes.csv project_directory/Output/collected_attributes_balanced.csv`

2.  Executing the script to get the class distribution  
    `./run.sh get-class-dist <INPUT CSV FILE> <OUTPUT DIR>`  

    Example Command  
    `./run.sh get-class-dist project_directory/Output/collected_attributes.csv`
