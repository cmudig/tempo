from flask import Blueprint, request, jsonify, redirect
from flask_login import current_user, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from tempo_server.compute.run import get_filesystem, get_demo_data_fs

def get_all_users(base_fs):
    if base_fs.exists("credentials.pkl"):
        return base_fs.read_file("credentials.pkl")
    return {}

def save_users(base_fs, users):
    base_fs.write_file(users, "credentials.pkl")
    
class User:
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
        
    def get_id(self):
        return self.user_id
    
    @classmethod
    def authenticate(cls, base_fs, user_id, password):
        all_credentials = get_all_users(base_fs)
        if user_id not in all_credentials:
            return None
        user = all_credentials[user_id]
        if check_password_hash(user["password_hash"], password):
            return User(user_id) # Can add info from the user object here
        return None
    
    @classmethod
    def get(cls, user_id):
        return User(user_id)
    
def create_user(base_fs, user_id, password):
    all_credentials = get_all_users(base_fs)
    if user_id in all_credentials:
        return None
    all_credentials[user_id] = { "password_hash": generate_password_hash(password) }
    save_users(base_fs, all_credentials)
    return User(user_id)

user_blueprint = Blueprint('users', __name__)

@user_blueprint.post("/login")
def login():
    user_id = request.json.get('user_id')
    password = request.json.get('password')
    remember = True if request.json.get('remember') else False
    user = User.authenticate(get_filesystem(), user_id, password)
    if user:
        login_user(user, remember=remember)
        return jsonify({"success": True, "user_id": user_id})
    else:
        return jsonify({"success": False, "error": 'The user ID and password you entered are invalid.'})

@user_blueprint.post("/create_user")
def signup_user():
    user_id = request.json.get('user_id')
    password = request.json.get('password')
    remember = True if request.json.get('remember') else False
    user = create_user(get_filesystem(), user_id, password)
    if user:
        login_user(user, remember=remember)
        demo_data_fs = get_demo_data_fs()
        if demo_data_fs:
            # Copy demo data into this user's directory
            demo_data_fs.copy_directory_contents(get_filesystem().subdirectory("datasets"))
        return jsonify({"success": True, "user_id": user_id})
    else:
        return jsonify({"success": False, "error": 'That username is taken.'})

@user_blueprint.get("/user_info")
@login_required
def get_user_info():
    return jsonify({"user_id": current_user.get_id()})

@user_blueprint.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")

def clear_all_users():
    base_fs = get_filesystem()
    base_fs.delete("credentials.pkl")
