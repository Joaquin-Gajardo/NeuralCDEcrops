# Repository for my master thesis at EPFL: "Neural controlled differential equations for Crops classification"

## Datasets
The datasets used for the experiments are available at request at jagajardo@uc.cl.

## Running the code
The code can be run from the command line. To run the code for either dataset change the data_root flag:

```
python crop_dataset_v10.py --data_root "YOUR/DATA/PATH" --no_logwandb
python swisscrop_dataset_v1.py --data_root "YOUR/DATA/PATH" --no_logwandb
```

To change the model pass the --model flag, options are: ncde, odernn, rnn, lstm, and gru.

For a complete list of CLI arguments check out:
```
python crop_dataset_v10.py --help
```
Process results stored in json files as summary plot or table:
```
python process_results.py --form table --metrics "test acc" "test f1"
```
