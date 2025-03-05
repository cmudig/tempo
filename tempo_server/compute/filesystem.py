import os
import json
import pickle
import tempfile
import shutil
import pandas as pd
import torch
from google.cloud import storage
from google.api_core import page_iterator

BINARY_FILE_TYPES = ('pickle', 'feather', 'bytes', 'pytorch')

FILE_TYPE_EXTENSIONS = {
    "txt": "str",
    "json": "json",
    "csv": "csv",
    "pickle": "pickle",
    "pkl": "pickle",
    "p": "pickle",
    "arrow": "feather",
    "feather": "feather",
    "zip": "bytes",
    "pth": "pytorch"
}

class LocalFilesystem:
    def __init__(self, base_path, readonly=False, temporary=False):
        self.base_path = base_path
        self.readonly = readonly
        self.temporary = temporary
        
    def __str__(self):
        return f"<LocalFilesystem '{self.base_path}'{', readonly' if self.readonly else ''}{', temporary' if self.temporary else ''}>"
        
    def list_files(self, *path):
        if not os.path.exists(os.path.join(self.base_path, *path)):
            return []
        return [x for x in os.listdir(os.path.join(self.base_path, *path))
                if not x.startswith(".")]
        
    def exists(self, *path):
        if not path: return os.path.exists(self.base_path)
        return os.path.exists(os.path.join(self.base_path, *path))
    
    def delete(self, *path):
        if not os.path.exists(os.path.join(self.base_path, *path)): return
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
            if format.lower() in ('str', 'bytes'):
                return file.read()
            elif format.lower() == 'json':
                return json.load(file, **kwargs)
            elif format.lower() == 'csv':
                return pd.read_csv(file, **kwargs)
            elif format.lower() == 'pickle':
                return pickle.load(file, **kwargs)
            elif format.lower() == 'feather':
                return pd.read_feather(file, **kwargs)
            elif format.lower() == 'pytorch':
                return torch.load(file, **kwargs)
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
            if format.lower() in ('str', 'bytes'):
                file.write(content)
            elif format.lower() == 'json':
                json.dump(content, file, **kwargs)
            elif format.lower() == 'csv':
                content.to_csv(file, **kwargs)
            elif format.lower() == 'pickle':
                pickle.dump(content, file, **kwargs)
            elif format.lower() == 'feather':
                content.to_feather(file, **kwargs)
            elif format.lower() == 'pytorch':
                torch.save(content, file)
            else:
                raise ValueError(f"Unsupported format {format}")
    
    def subdirectory(self, *path):
        return LocalFilesystem(os.path.join(self.base_path, *path), self.readonly)
    
    def copy_directory_contents(self, dest_fs):
        """Copies the contents of the tree rooted in this object to the given destination filesystem."""
        if not isinstance(dest_fs, LocalFilesystem):
            raise ValueError(f"Unknown destination filesystem {dest_fs}")
        shutil.copytree(self.base_path, dest_fs.base_path)
        
    def copy_file(self, dest_fs, *path):
        """Copies the item at the given path (last arguments) to the given destination.
        Note that the destination comes first."""
        print("Copying", os.path.join(self.base_path, *path), dest_fs)
        if isinstance(dest_fs, LocalFilesystem):
            if not dest_fs.exists():
                os.makedirs(dest_fs.base_path)
            shutil.copyfile(os.path.join(self.base_path, *path), os.path.join(dest_fs.base_path, path[-1]))
        else:
            content = self.read_file(*path)
            dest_fs.write_file(content, path[-1])
        
    def rename(self, dest_fs):
        """Moves the contents of the tree rooted in this object to the given destination filesystem,
        and returns that filesystem."""
        if isinstance(dest_fs, LocalFilesystem):
            shutil.move(self.base_path, dest_fs.base_path)
        else:
            self.copy_directory_contents(dest_fs)
            self.delete()
        return dest_fs
        
    def make_temporary_directory(self):
        tempdir = tempfile.TemporaryDirectory()
        return LocalFilesystem(tempdir.name, temporary=True)
    
    def __del__(self):
        if self.temporary and self.exists():
            shutil.rmtree(self.base_path)
            
def _item_to_value(iterator, item):
    return item

def list_directories(bucket_name, prefix):
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x for x in iterator]

class GCSFilesystem:
    def __init__(self, client, bucket_name, base_path='', local_fallback=None, readonly=False, temporary=False):
        self.client = client
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)
        self.base_path = base_path
        self.readonly = readonly
        self.temporary = temporary
        if local_fallback is not None:
            self.local_fallback = LocalFilesystem(local_fallback) if isinstance(local_fallback, str) else local_fallback
        else:
            self.local_fallback = None
        
    def __str__(self):
        return f"<GCSFilesystem '{self.bucket_name}' -> '{self.base_path}'{', readonly' if self.readonly else ''}{', temporary' if self.temporary else ''}>"
        
    def get_local_fallback(self, *path):
        if self.local_fallback:
            return self.local_fallback.subdirectory(*self.base_path.strip('/').split('/'), *path)
        return None
    
    def list_files(self, *path):
        prefix = self.make_full_path(*path)
        return sorted([
            *(os.path.basename(b.name) 
              for b in self.bucket.list_blobs(prefix=prefix)
              if os.path.dirname(b.name) == prefix),
            *(os.path.basename(p.rstrip('/')) for p in list_directories(self.bucket_name, prefix))
        ])
        
    def make_full_path(self, *path):
        return '/'.join((self.base_path, *path)).lstrip('/')
        
    def exists(self, *path):
        full_path = self.make_full_path(*path)
        # check if blob exists
        if self.bucket.blob(full_path).exists(): return True
        # check if it's a directory
        if next(self.bucket.list_blobs(prefix=full_path), None) is not None:
            return True
        return False
    
    def delete(self, *path):
        full_path = self.make_full_path(*path)
        
        # Delete if a file
        blob = self.bucket.blob(full_path)
        if blob.exists():
            blob.reload()
            generation_match_precondition = blob.generation

            blob.delete(if_generation_match=generation_match_precondition)

        # Delete if a subdirectory
        for blob in self.bucket.list_blobs(prefix=full_path):
            blob.delete()
            
    def read_file(self, *path, format=None, **kwargs):
        if format is None: 
            try:
                format = FILE_TYPE_EXTENSIONS[os.path.splitext(path[-1])[-1].replace('.', '')]
            except:
                raise ValueError(f"Unknown format for path {path[-1]}")
        full_path = self.make_full_path(*path)
        with self.bucket.blob(full_path).open('rb' if format in BINARY_FILE_TYPES else 'r') as file:
            if format.lower() in ('str', 'bytes'):
                return file.read()
            elif format.lower() == 'json':
                return json.load(file, **kwargs)
            elif format.lower() == 'csv':
                return pd.read_csv(file, **kwargs)
            elif format.lower() == 'pickle':
                return pickle.load(file, **kwargs)
            elif format.lower() == 'feather':
                return pd.read_feather(file, **kwargs)
            elif format.lower() == 'pytorch':
                return torch.load(file, **kwargs)
            raise ValueError(f"Unsupported format {format}")
        
    def write_file(self, content, *path, format=None, **kwargs):            
        if format is None: 
            try:
                format = FILE_TYPE_EXTENSIONS[os.path.splitext(path[-1])[-1].replace('.', '')]
            except:
                raise ValueError(f"Unknown format for path {path[-1]}")
            
        full_path = self.make_full_path(*path)
        with self.bucket.blob(full_path).open('wb' if format in BINARY_FILE_TYPES else 'w') as file:
            if format.lower() in ('str', 'bytes'):
                file.write(content)
            elif format.lower() == 'json':
                json.dump(content, file, **kwargs)
            elif format.lower() == 'csv':
                content.to_csv(file, **kwargs)
            elif format.lower() == 'pickle':
                pickle.dump(content, file, **kwargs)
            elif format.lower() == 'feather':
                content.to_feather(file, **kwargs)
            elif format.lower() == 'pytorch':
                torch.save(content, file)
            else:
                raise ValueError(f"Unsupported format {format}")
    
    def subdirectory(self, *path):
        return GCSFilesystem(self.client, self.bucket_name, self.make_full_path(*path), local_fallback=self.local_fallback, readonly=self.readonly)
    
    def copy_directory_contents(self, dest_fs):
        """Copies the contents of the tree rooted in this object to the given destination filesystem."""
        for blob in self.bucket.list_blobs(prefix=self.base_path):
            self.copy_file(dest_fs, blob.name.replace(self.base_path, '').lstrip('/'))
        
    def copy_file(self, dest_fs, *path):
        """Copies the item at the given path (last arguments) to the given destination.
        Note that the destination comes first."""
        print("Copying", self.make_full_path(*path), dest_fs)
        if isinstance(dest_fs, GCSFilesystem):
            self.bucket.copy_blob(self.bucket.blob(self.make_full_path(*path)), 
                                dest_fs.bucket,
                                dest_fs.make_full_path(path[-1]))
        else:
            content = self.read_file(*path)
            dest_fs.write_file(content, path[-1])
        
    def rename(self, dest_fs):
        """Moves the contents of the tree rooted in this object to the given destination filesystem,
        and returns that filesystem."""
        if isinstance(dest_fs, GCSFilesystem):
            self.copy_directory_contents(dest_fs)
            self.delete()
        else:
            self.copy_directory_contents(dest_fs)
        return dest_fs
        
    def make_temporary_directory(self):
        tempdir = tempfile.TemporaryDirectory()
        return LocalFilesystem(tempdir.name, temporary=True)