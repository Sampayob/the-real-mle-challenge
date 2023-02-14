# The NY Estimator Problem - Solution

## Challenge 1 - Refactor DEV code
The code inside `01-exploratory-data-analysis.ipynb` and `02-explore-classifier-model.ipynb` has been refactored and tested: 
- `\config` directory:
    - `confing.py`: script to manage and organize all directories and filepaths for easy and centralized access.
    - `args.json`: JSON file which contain different parameters used in data preprocess and model training, evaluation and prediction to ensure reproducibility.
- `\src` directory and subdirectories (`\data`, `\models`, `\utils`,...) to organize all scripts except `config.py`:
    - `data\preprocess.py`: script to preprocess the data from `\raw` to `\processed` as in `01-exploratory-data-analysis.ipynb`.
        - Used `map` instead of `apply` when applying functions to DataFrame cols (Example: `num_bathroom_from_text`).
        - Grouped the different feature transformations in one function: `feature_engineering()`.
    - `models\main.py`: to launch `train_model`, `evaluation_model` and `predict` functions, which respectively call ``models\train.py`, `models\evaluation.py`, `models\predict.py` scripts:
        - `train.py`: prepara de processed data (encoding categorical features, split data in train-test...) and train the classification model.
        - `evaluation.py`: contains different functions to get metrics and plots to evaluate the trained model (accuracy, ROC AUC, confussion matrix...).
        - `predict.py`: preprocess input data to make a prediction loading the previously trained model.
- `\tests` directory to manage tests:
    - `test_data.py`: test data preprocess to generate `preprocessed_listings.csv`.
    - `test_train.py`: test model training methods launched from `src\models\main.py` as well as evaluation methods.
    - `test_predict.py`: test model predict method launched from `src\models\main.py` passing some input data to return a prediction.
- `\utils` directory:
    - `utils.py`: script which contains helper functions.


## Challenge 2 - Build an API
- Created a Flask API to serve the model inside the `api` directory:
    - `app.py`: script from which the Flask app is launched. It contains a GET method (`/`) and a POST method (`/predict_category`) which use the previously trained model to make a predict based on new input data.
    - `prediction.py`: script which have the modules to process an inference and return a prediction. 
    - `simple_classifier.pkl`: classification model previously trained. It used only the subset of features signaled in `02-explore-classifier-model.ipynb`: `FEATURE_NAMES`.
    - `args.json`: same set of dictionaries as in `config\` which contain information for the inference preprocess to improve reproducibility.
    - `tests\test.py`: test the two methods present in the Flask app `app.py`.
        
    *I had to replicate some previous code in the script `prediction.py` and move the model (`simple_classifier.pkl`) and the arguments json (`args.json`) which contains information for the inference preprocess because `ModuleNameError` while importing directories and modules like `src` or `config`.

- The API was tested making a GET and a POST request to `0.0.0.0:8080/` through POSTMAN with one data input in JSON format.


## Challenge 3 - Dockerize your solution
- The Flask API can be deployed on a Docker container previously building a Docker image based on the Dockerfile in `/api/`. The commands to build the image and launch de Docker container from the `/api/` directory are:
    1. Build Docker image: `docker build -t ny-estimator .`
    2. Launch Docker container (first time): `docker build -d -p 8080:8080 --name ny-estimator-container ny-estimator`
    3. Launch Docker container (subsequent times): `docker build -d -p 8080:8080 ny-estimator`

    *If want to get the API response in real time like launching `app.py`: change `-d` flag with `-it` flag.

- The API was tested as in the previous challenge.
