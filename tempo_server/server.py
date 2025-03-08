from flask import Flask, send_from_directory, request, jsonify, redirect, render_template
from werkzeug.serving import is_running_from_reloader
import os
import atexit
import argparse
from werkzeug.middleware.profiler import ProfilerMiddleware
from tempo_server.blueprints.data import data_blueprint
from tempo_server.blueprints.models import models_blueprint
from tempo_server.blueprints.datasets import datasets_blueprint
from tempo_server.blueprints.tasks import tasks_blueprint
from tempo_server.blueprints.slices import slices_blueprint
from tempo_server.blueprints.user import User, create_user
from tempo_server.compute.run import setup_worker, make_filesystem_from_info
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect

FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client", "dist")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='Tempo',
                description='Start a Flask server to create, interpret, and compare time-series prediction model formulations.')

    parser.add_argument('--local-data', type=str, help='Path to local directory containing a "datasets" directory which contains tempo datasets. Can also be passed via the TEMPO_LOCAL_DATA environment variable.')
    parser.add_argument('--port', type=int, default=4999, help='Port to run the server on')
    parser.add_argument('--public', action='store_true', default=False, help='Open the server to public network traffic')
    parser.add_argument('--profile', action='store_true', default=False, help='Print cProfile performance logs for each API call')
    parser.add_argument('--single-thread', action='store_true', default=False, help='Disable multithreading for slice finding')
    parser.add_argument('--users', action='store_true', default=False, help='Partition datasets by users and require login')
    parser.add_argument('--debug', action='store_true', default=False, help='Print detailed logging messages')
    parser.add_argument('--gcs-bucket', type=str, default=None, help='If provided, store models and data in this GCS bucket (assumed google auth credentials are defined in an env variable). Option can also be passed via the TEMPO_GCS_BUCKET environemnt variable.')
    
    args = parser.parse_args()
    public = args.public or os.environ.get("TEMPO_PRODUCTION") == "1"
    local_data_dir = args.local_data or os.environ.get("TEMPO_LOCAL_DATA") or os.path.dirname(os.path.dirname(__file__))

    gcs_bucket = args.gcs_bucket or os.environ.get("TEMPO_GCS_BUCKET")
    if gcs_bucket:
        fs_info = {'type': 'gcs', 'bucket': gcs_bucket, 'local_fallback': local_data_dir}
    else:
        if not args.local_data:
            raise ValueError("Must provide a local data directory (--local-data command-line argument) if not using GCS")
        fs_info = {'type': 'local', 'path': args.local_data}
    base_fs = make_filesystem_from_info(fs_info)
    
    app = Flask(__name__, template_folder=FRONTEND_BUILD_DIR)
    csrf = CSRFProtect(app)
    if not args.users:
        app.config['WTF_CSRF_CHECK_DEFAULT'] = False

    partition_by_user = args.users
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
    
    # Path for our main Svelte page
    @app.route("/")
    def client():
        return render_template('index.html')

    # Route to add static files (CSS and JS)
    @app.route("/<path:path>")
    def base(path):
        return send_from_directory('../client/dist', path)
    
    @app.post("/login")
    def login():
        user_id = request.json.get('user_id')
        password = request.json.get('password')
        remember = True if request.json.get('remember') else False
        user = User.authenticate(base_fs, user_id, password)
        if user:
            login_user(user, remember=remember)
            return jsonify({"success": True, "user_id": user_id})
        else:
            return jsonify({"success": False, "error": 'The user ID and password you entered are invalid.'})

    @app.post("/create_user")
    def signup_user():
        user_id = request.json.get('user_id')
        password = request.json.get('password')
        remember = True if request.json.get('remember') else False
        user = create_user(base_fs, user_id, password)
        if user:
            login_user(user, remember=remember)
            return jsonify({"success": True, "user_id": user_id})
        else:
            return jsonify({"success": False, "error": 'That username is taken.'})

    @app.get("/user_info")
    @login_required
    def get_user_info():
        return jsonify({"user_id": current_user.get_id()})
    
    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect("/")
    
    @login_manager.user_loader
    def load_user(user_id):
        if not args.users: return None
        return User.get(user_id)

    @login_manager.unauthorized_handler
    def unauthorized_callback():
        return "You must be logged in to access this content.", 403

    @app.route("/logs")
    def logs():
        log_dir = os.path.join(local_data_dir, "logs")
        if os.path.exists(os.path.join(log_dir, "log.txt")):
            with open(os.path.join(log_dir, "log.txt"), "r") as file:
                return f"""
            <html>
                <head><title>Tempo Logs</title></head>
                <body>
                <pre>{file.read()}</pre>
                </body>
            </html>
            """
                
    if args.profile:
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, sort_by=["cumtime"], restrictions=[100])
        
    worker = None
    
    if public or is_running_from_reloader():
        print("Starting a worker")
        log_dir = os.path.join(local_data_dir, "logs")
        if not os.path.exists(log_dir):
            os.mkdir(os.path.join(log_dir))
        worker = setup_worker(fs_info, log_dir, verbose=args.debug, partition_by_user=partition_by_user)
        worker.start()
    
    def close_running_threads():
        if worker is not None:
            worker.terminate()
            print("Shut down model worker")
    atexit.register(close_running_threads)
    
    app.run(debug=not args.public, port=args.port, host='0.0.0.0' if public else '127.0.0.1', threaded=not args.single_thread)
