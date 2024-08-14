'''
Run instructions:
from the parent tempo repo directory, run python -m tempo_server.tests.test_models <path/to/datasets>.
'''
from ..compute.model import Model
from ..compute.dataset import Dataset
from ..compute.filesystem import LocalFilesystem
import sys

if __name__ == '__main__':
    dataset_name = 'ecommerce_new'
    model_name = 'Future Purchase 3'
    
    # get the filesystem path
    fs = LocalFilesystem(sys.argv[1])
    
    # load the dataset
    dataset = Dataset(fs.subdirectory(dataset_name))
    if not dataset.fs.exists():
        print("Dataset does not exist")
        exit(1)
    dataset.load_if_needed()

    # fetch the model with the given name
    model = dataset.get_model(model_name)
    spec = model.get_spec()

    def update_fn(progress):
        print("Progress:", progress)
    # run the training function, and print progress using update_fn
    model.make_model(dataset, spec, update_fn=update_fn)

    print("Results are saved in:", model.result_fs)