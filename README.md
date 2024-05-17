# thesis-web-app

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
