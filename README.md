# thesis-web-app

## About
This repository contains the files for my thesis titled "Enhanced Aerial Image Segmentation for Land Cover Detection Using UNetFormer++: Integration in Nested UNet Models and Transformers". It includes everything necessary to run both the frontend and backend of the web app.

The web app features three main pages:

- Predict Page: Users can upload an image to receive its semantic segmentation using various models trained with the [training script](https://github.com/nobertpr/thesis-training-script).
- History Page: Users can view, download, or delete previously segmented images and their corresponding semantic segmentations.
- Maps Page: The semantic segmentation mask is overlaid on a Google Earth Engine Map for visual analysis.

## [Live Demo](https://huggingface.co/spaces/nobertpr/Thesis-LandCoverAI)
History page is not available due to limitation of hugging face spaces for db file.

## Requirements
You need to install all required packages which are listed in the requirements.txt to run this web app.

Use the package manager pip to install all the required package.

`pip install -r requirements.txt`

## About
This is a repository containing the web app used for my thesis. The app is created using streamlit for the frontend and fastapi for the backend.

Create two separate directories for the front end and backend

To run frontend, go into the frontend directory and run the following:
`streamlit run 1_🔎_Predict.py`

To run backend, go into the backend directory and run the following:
`uvicorn main:app --reload`

## Note
The models, some predicted images and maps are not included due to limitation of file size in github.
