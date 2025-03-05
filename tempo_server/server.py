from flask import Flask, send_from_directory, request, jsonify, send_file
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
from tempo_server.compute.run import setup_worker
from tempo_server.compute.filesystem import LocalFilesystem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='Tempo',
                description='Start a Flask server to create, interpret, and compare time-series prediction model formulations.')

    parser.add_argument('base_path', type=str)
    parser.add_argument('--port', type=int, default=4999, help='Port to run the server on')
    parser.add_argument('--public', action='store_true', default=False, help='Open the server to public network traffic')
    parser.add_argument('--profile', action='store_true', default=False, help='Print cProfile performance logs for each API call')
    parser.add_argument('--single-thread', action='store_true', default=False, help='Disable multithreading for slice finding')
    parser.add_argument('--debug', action='store_true', default=False, help='Print detailed logging messages')
    
    args = parser.parse_args()

    app = Flask(__name__)
    app.register_blueprint(data_blueprint)
    app.register_blueprint(models_blueprint)
    app.register_blueprint(datasets_blueprint)
    app.register_blueprint(tasks_blueprint)
    app.register_blueprint(slices_blueprint)
    
    # Path for our main Svelte page
    @app.route("/")
    def client():
        return send_from_directory('../client/dist', 'index.html')

    # Route to add static files (CSS and JS)
    @app.route("/<path:path>")
    def base(path):
        return send_from_directory('../client/dist', path)
    
    @app.route("/logs")
    def logs():
        log_dir = os.path.join(args.base_path, "logs")
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
    
    if os.environ.get("TEMPO_PRODUCTION") == "1" or is_running_from_reloader():
        print("Starting a worker")
        log_dir = os.path.join(args.base_path, "logs")
        if not os.path.exists(log_dir):
            os.mkdir(os.path.join(log_dir))
        worker = setup_worker(LocalFilesystem(args.base_path), log_dir, verbose=args.debug)
        worker.start()
    
    def close_running_threads():
        if worker is not None:
            worker.terminate()
            print("Shut down model worker")
    atexit.register(close_running_threads)
    
    app.run(debug=not args.public, port=args.port, host='0.0.0.0' if args.public else '127.0.0.1', threaded=not args.single_thread)