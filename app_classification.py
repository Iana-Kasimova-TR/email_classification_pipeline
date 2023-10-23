import os

import fasttext
import joblib
import numpy as np
import yaml
from flask import Flask, jsonify, render_template, request, send_from_directory

from src.load_parameters import read_params

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


class NotEmpty(Exception):
    def __init__(self, message="Text shouldn't be empty"):
        self.message = message
        super().__init__(self.message)


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = fasttext.load_model(model_dir_path)
    prediction = model.predict(data)[0][0].replace("__label__", "")
    return prediction


def validate_input(dict_request):
    text = dict_request["email_text"]
    if text == "":
        raise NotEmpty()
    else:
        return True


def form_response(dict_request):
    try:
        if validate_input(dict_request):
            data = dict_request["email_text"]
            response = predict(data)
            return response
    except NotEmpty as e:
        response = str(e)
        return response


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def index():
    try:
        if request.form:
            dict_req = dict(request.form)
            response = form_response(dict_req)
            return render_template("index.html", response=response)
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong!! Try again later!"}
        error = {"error": e}
        return render_template("404.html", error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
