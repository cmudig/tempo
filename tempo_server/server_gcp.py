from flask import Flask, send_from_directory, request, jsonify, redirect, render_template
from werkzeug.serving import is_running_from_reloader
import os
import atexit
from tempo_server.blueprints.data import data_blueprint
from tempo_server.blueprints.models import models_blueprint
from tempo_server.blueprints.datasets import datasets_blueprint
from tempo_server.blueprints.tasks import tasks_blueprint
from tempo_server.blueprints.slices import slices_blueprint
from tempo_server.blueprints.user import User, user_blueprint
from tempo_server.compute.run import setup_worker, make_filesystem_from_info, get_filesystem
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect

FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client", "dist")

public = os.environ.get("TEMPO_PRODUCTION") == "1"
local_data_dir = os.environ.get("TEMPO_LOCAL_DATA") or os.path.dirname(__file__)

gcs_bucket = os.environ.get("TEMPO_GCS_BUCKET")
if gcs_bucket:
    fs_info = {'type': 'gcs', 'bucket': gcs_bucket}
else:
    if not local_data_dir:
        raise ValueError("Must provide a local data directory (--local-data command-line argument) if not using GCS")
    fs_info = {'type': 'local', 'path': local_data_dir}
base_fs = make_filesystem_from_info(fs_info)
demo_data_dir = os.environ.get("TEMPO_DEMO_DATA", None)
if demo_data_dir:
    demo_data_fs = base_fs.subdirectory(demo_data_dir)
    assert demo_data_fs.exists(), f"Demo data must exist in {base_fs}"
    fs_info['demo_data'] = demo_data_dir

app = Flask(__name__, template_folder=FRONTEND_BUILD_DIR)
csrf = CSRFProtect(app)

partition_by_user = True
app.config['LOGIN_DISABLED'] = os.environ.get("LOGIN_DISABLED") == "1" or not partition_by_user

# Read secret key from secret.txt if available, otherwise fallback (dev only)
if os.path.exists("secret.txt"):
    with open("secret.txt", "r") as file:
        app.secret_key = file.read().strip()
else:
    print("WARNING: Using the development secret key. If using in production, please make sure a secret.txt file is present.")
    app.secret_key = "efd06fa9d66fbdc84025a05066bc85d337e1c89a9cbff62ab4cf6fc6c5077f50"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "/"

app.register_blueprint(data_blueprint)
app.register_blueprint(models_blueprint)
app.register_blueprint(datasets_blueprint)
app.register_blueprint(tasks_blueprint)
app.register_blueprint(slices_blueprint)
app.register_blueprint(user_blueprint)

# Path for our main Svelte page
@app.route("/")
def client():
    return render_template('index.html')

# Route to add static files (CSS and JS)
@app.route("/<path:path>")
def base(path):
    return send_from_directory('../client/dist', path)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@login_manager.unauthorized_handler
def unauthorized_callback():
    return "You must be logged in to access this content.", 403

worker = None

def close_running_threads():
    if worker is not None:
        worker.terminate()
        print("Shut down model worker")

if public or is_running_from_reloader():
    print("Starting a worker")
    log_dir = None
    worker = setup_worker(fs_info, log_dir, verbose=False, partition_by_user=partition_by_user)
    worker.start()
    atexit.register(close_running_threads)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0' if public else '127.0.0.1', threaded=os.environ.get("TEMPO_SINGLE_THREAD") != "1")
