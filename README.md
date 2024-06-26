# Time Series Model Selection Tool

## Setup Instructions

To set up, `cd` into the main repository directory, and run
`pip install -r requirements.txt`. Then `cd` into the `client` directory and run
`npm install && npm run build`.

Create a `tempo-data` directory somewhere on the machine, with a directory called
`datasets` inside that. Place any datasets you'd like to load within that directory.
(The `tempo-data` directory can be anything that includes a `datasets` subdirectory, 
including the main Tempo repo folder.)

To run the tool, `cd` into the main directory and run `python -m tempo_server.server <path to tempo-data>`. 
Then visit `http://127.0.0.1:4999` in your browser to see the tool.

To automatically rebuild the client while developing, `cd` into the `client`
directory and run `npm run autobuild`. After making changes to the source code, 
you will need to reload the browser page to see the updated version.