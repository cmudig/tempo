import os
import json
import pickle
import tempfile
import shutil
import pandas as pd

BINARY_FILE_TYPES = ('pickle', 'feather')

FILE_TYPE_EXTENSIONS = {
    "txt": "str",
    "json": "json",
    "csv": "csv",
    "pickle": "pickle",
    "pkl": "pickle",
    "p": "pickle",
    "arrow": "feather",
    "feather": "feather"
}

class LocalFilesystem:
    def __init__(self, base_path, readonly=False, temporary=False):
        self.base_path = base_path
        self.readonly = readonly
        self.temporary = temporary
        
    def list_files(self, *path):
        return [x for x in os.listdir(os.path.join(self.base_path, *path))
                if not x.startswith(".")]
        
    def exists(self, *path):
        if not path: return os.path.exists(self.base_path)
        return os.path.exists(os.path.join(self.base_path, *path))
    
    def delete(self, *path):
        if not path: shutil.rmtree(self.base_path)
        else:
            fp = os.path.join(self.base_path, *path)
            if os.path.isdir(fp):
                shutil.rmtree(fp)
            else:
                os.remove(fp)
            
    def read_file(self, *path, format=None, **kwargs):
        if format is None: 
            try:
                format = FILE_TYPE_EXTENSIONS[os.path.splitext(path[-1])[-1].replace('.', '')]
            except:
                raise ValueError(f"Unknown format for path {path[-1]}")
        with open(os.path.join(self.base_path, *path), 'rb' if format in BINARY_FILE_TYPES else 'r') as file:
            if format.lower() == 'str':
                return file.read()
            elif format.lower() == 'json':
                return json.load(file, **kwargs)
            elif format.lower() == 'csv':
                return pd.read_csv(file, **kwargs)
            elif format.lower() == 'pickle':
                return pickle.load(file, **kwargs)
            elif format.lower() == 'feather':
                return pd.read_feather(file, **kwargs)
            raise ValueError(f"Unsupported format {format}")
        
    def write_file(self, content, *path, format=None, **kwargs):
        dest_path = os.path.join(self.base_path, *path)
        if not os.path.exists(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path))
            
        if format is None: 
            try:
                format = FILE_TYPE_EXTENSIONS[os.path.splitext(path[-1])[-1].replace('.', '')]
            except:
                raise ValueError(f"Unknown format for path {path[-1]}")
            
        with open(dest_path, 'wb' if format in BINARY_FILE_TYPES else 'w') as file:
            if format.lower() == 'str':
                file.write(content)
            elif format.lower() == 'json':
                json.dump(content, file, **kwargs)
            elif format.lower() == 'csv':
                content.to_csv(file, **kwargs)
            elif format.lower() == 'pickle':
                pickle.dump(content, file, **kwargs)
            elif format.lower() == 'feather':
                content.to_feather(file, **kwargs)
            else:
                raise ValueError(f"Unsupported format {format}")
    
    def subdirectory(self, *path):
        return LocalFilesystem(os.path.join(self.base_path, *path), self.readonly)
    
    def copy_directory_contents(self, dest_fs):
        """Transfers the contents of the tree rooted in this object to the given destination filesystem."""
        if not isinstance(dest_fs, LocalFilesystem):
            raise ValueError(f"Unknown destination filesystem {dest_fs}")
        shutil.copytree(self.base_path, dest_fs.base_path)
        
    def make_temporary_directory(self):
        tempdir = tempfile.TemporaryDirectory()
        return LocalFilesystem(tempdir.name, temporary=True)
    
    def __del__(self):
        if self.temporary and self.exists():
            shutil.rmtree(self.base_path)