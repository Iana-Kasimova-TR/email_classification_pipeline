import os

import fasttext
from flask import Flask, jsonify, render_template, request

import src.database.db_handler as db_handler
import src.utils.log_config as log_config
from src.utils.load_parameters import read_params

webapp_root = os.getenv("APP_ROOT", "webapp")
params_path = os.getenv("PARAMS_PATH", "params/params.yaml")

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
db_uri = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["SQLALCHEMY_DATABASE_URI"] = db_uri

db_handler.init_db(app)
log_config.setup_logging(app)

health_status = True


class NotEmpty(Exception):
    def __init__(self, message="Text shouldn't be empty"):
        self.message = message
        super().__init__(self.message)


def predict(data) -> str:
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = fasttext.load_model(model_dir_path)
    prediction = model.predict(data)[0][0].replace("__label__", "")
    app.logger.info(f"Prediction: {prediction}")
    if db_handler.save_to_database(data, prediction):
        app.logger.info("Prediction saved in database")
    else:
        app.logger.error("Prediction not saved in database")
    return prediction


def validate_input(dict_request) -> bool:
    text = dict_request["email_text"]
    if text == "":
        app.logger.error("Text is empty")
        raise NotEmpty()
    else:
        return True


def form_response(dict_request) -> str:
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


@app.route("/health", methods=["GET"])
def health():
    if health_status:
        resp = jsonify(health="healthy")
        resp.status_code = 200
    else:
        resp = jsonify(health="unhealthy")
        resp.status_code = 500   
    return resp


@app.route("/model_version", methods=["GET"])
def get_model_version():
    try:
        model_name = os.getenv("YOUR_MODEL_NAME")
        model_version = os.getenv("YOUR_MODEL_VERSION")
        resp = jsonify(model_name=model_name, model_version=model_version)
        resp.status_code = 200
        app.logger.info(f"model version: {model_version}")
        return resp
    except Exception as e:
        error = {"error": e}
        return render_template("error.html", error=error)


@app.route("/", methods=["POST"])
def index():
    try:
        if request.form:
            dict_req = dict(request.form)
            response = form_response(dict_req)
            app.logger.info(f"Response: {response}")
            return render_template(
                "index.html", response=response, email_text=dict_req["email_text"]
            )
    except Exception as e:
        error = {"error": e}
        return render_template("error.html", error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
