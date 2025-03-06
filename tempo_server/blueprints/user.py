from werkzeug.security import generate_password_hash, check_password_hash

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
