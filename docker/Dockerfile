FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV SQLALCHEMY_DATABASE_URI="sqlite:///email.db"
ENV PARAMS_PATH="params/params.yaml"
ENV APP_ROOT="webapp"
ENV YOUR_MODEL_NAME="fasttext_model"
ENV YOUR_MODEL_VERSION="Production"
ARG MODEL_DIR="/app/webapp/model_webapp_dir/"


RUN pip install poetry
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock /app/

RUN pip install fasttext-wheel 
RUN pip install gunicorn
RUN poetry install 

COPY app_classification.py /app/app_classification.py
COPY params /app/params
COPY webapp /app/webapp
#COPY email.db /app/email.db
COPY src /app/src

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

RUN poetry run mlflow artifacts download -u models:/$YOUR_MODEL_NAME/$YOUR_MODEL_VERSION -d $MODEL_DIR

EXPOSE 5050

CMD ["gunicorn", "app_classification:app", "-b", "0.0.0.0:5050"]
