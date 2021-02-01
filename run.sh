#!/bin/bash

echo "$1"

if [ "$1" = "preprocess-text" ]; then
    python preprocessing/preprocessing_text.py $2 $3
elif [ "$1" = "preprocess-attributes" ]; then
    python preprocessing/preprocessing_attributes.py $2 $3 $4 $5 $6
elif [ "$1" = "preprocess-text-2" ]; then
    python preprocessing/preprocessing_reviews.py $2 $3
elif [ "$1" = "preprocess-text-attr" ]; then
    python preprocessing/preprocessing_review_attributes.py $2 $3 $4
elif [ "$1" = "train-image-model" ]; then
    python image_classifier/train_image_model.py $2
elif [ "$1" = "train-attribute-model" ]; then
    python attribute_classifier/train_attribute_model.py $2 $3 $4 $5
elif [ "$1" = "train-text-model" ]; then
    python text_classifier/train_text_model.py $2
elif [ "$1" = "test-text-attr" ]; then
    python multimodal_classifiers/test_review_attributes.py $2 $3
elif [ "$1" = "test-image-attr" ]; then
    python multimodal_classifiers/test_images_attributes.py $2 $3 $4
elif [ "$1" = "test-image-text" ]; then
    python multimodal_classifiers/test_images_reviews.py $2 $3
elif [ "$1" = "test-text-img-attr" ]; then
    python multimodal_classifiers/test_images_review_attributes.py $2 $3 $4
elif [ "$1" = "test-baseline" ]; then
    python baseline_scripts/evaluate_baseline.py $2 $3 $4 $5
elif [ "$1" = "balance-data" ]; then
    python helpers/balance_data.py $2 $3
elif [ "$1" = "get-class-dist" ]; then
    python helpers/get_class_distribution.py $2 $3
else
    echo "Invalid arguments"
fi

