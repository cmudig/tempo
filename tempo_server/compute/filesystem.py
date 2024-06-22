import os
import json
import pickle

class LocalFilesystem:
    def __init__(self, base_path, readonly=False):
        self.base_path = base_path
        self.readonly = readonly
        
    def read_file(self, *path, format='str'):
        with open(os.path.join(self.base_path, *path), 'r') as file:
            if format.lower() == 'str':
                return file.read()
            elif format.lower() == 'json':
                return json.load(file)
            elif format.lower() == 'pickle':
                return pickle.load(file)
            raise ValueError(f"Unsupported format {format}")
        
    def write_file(self, content, *path, format='str'):
        with open(os.path.join(self.base_path, *path), 'w') as file:
            if format.lower() == 'str':
                file.write(content)
            elif format.lower() == 'json':
                json.dump(content, file)
            elif format.lower() == 'pickle':
                pickle.dump(content, file)
            raise ValueError(f"Unsupported format {format}")
    
    def subdirectory(self, *path):
        return LocalFilesystem(os.path.join(self.base_path, *path), self.readonly)