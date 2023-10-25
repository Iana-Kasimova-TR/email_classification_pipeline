# Email Classification Pipeline
A project for deploying a model, that predicts the class of compliant email
Also contains scripts for preprocessing, training and evaluation model


## Installation
For running this project, need to be installed Docker and the next things:
```bash
pip install mlflow \
pip install makefile
```

## Installation
For running tests:
```bash
make test
```
For building a solution:
```bash
make build
```
For running solution:
```bash
make run
```
## Usage
Navigate to the 
```
http://localhost:5050/model_version
```
Then put the text of email into text box and click on button 'Predict'

You also can check the version of model by endpoint '/model_verrsion'


## Notes:
### What to improve: 
- Deploy stage in make should be changed for deploying not locally
- .toml file divide for dev and prod env
- replace print() on logger for better tracking
- extract logic of working with db in different layer, so that we can change the type of db easily, without making changes in core
- increase test coverage
- improve code quality, fix error based on report of flake8



### Usage of ML pipeline:
For running ML pipeline we need:
- clone git repository [repo](https://github.com/Iana-Kasimova-TR/email_classification_pipeline/tree/dvc_mlflow_pipe)
- install dvc - [DVC](https://dvc.org)
- install poetry - [Poetry](https://python-poetry.org/docs/)
- install mlflow - [MLFlow](https://mlflow.org/docs/latest/quickstart.html)
- run 
```bash
mlflow ui
```
- run from the root of repo, before download 'emails.csv' dataset - now I keep it locally, but it can be located in a cloud 
```bash
dvc add data/external/emails.csv
```
- run
```bash
dvc repro
```
- have a look on experiments in mlflow ui


### Monitor data and model driff
For monitor drift you can use evidently library and run
```bash
python src/report/model_monitoring.py
```

if it fails, try to run 
```bash
export PYTHONPATH='{absolute_path_to_project}/email_classification_pipeline/src'
```


