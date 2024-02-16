# Time Series Model Selection Tool

## Installation

To set up, `cd` into the main `timeseries_model_selection` directory, and run
`pip install -r requirements.txt`. Then `cd` into the `client` directory and run
`npm install`.

Create a `datasets` directory in the top level of the repo folder, and place any
datasets you'd like to load within that directory.

To run the tool, `cd` into the main directory and run `python server.py datasets/<dataset_name>`. 
In a separate terminal, `cd` into the `client` directory and run `npm run autobuild`.
Then visit `http://127.0.0.1:4999` in your browser to see the tool.

After making changes to the source code, you will need to reload the browser page
to see the updated version.