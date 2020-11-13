# AEs-VAEs-TDA (WIP)

## Installation 


### pre-commit
Pre-commit is used to protect the master branch from (accidental) commits. 

```bash
pip install pre-commit
pre-commit --version
pre-commit install
```

### torchph 
https://github.com/c-hofer/torchph

Recommended to install it this way: (not with conda)
```bash
git clone https://github.com/c-hofer/torchph.git
cd torchph
python setup.py develop
```

### dependencies
```bash
pip3 install -r requirements.txt
```
## Usage

The simulator can be used with:
```bash
python main.py
```

One can choose a model, the number of parallel processes and a configuration for the model.
Please refer to scripts/config_library/sample.py, where sample configurations can be found.
Currently the simulator supports:
- TopoAE (command: -m 'topoae')
- WCAE (command: -m 'WCAE')
- TSNE, UMAP (command: -m 'competitor')

For the competitors, the model is defined directly in the configuration file.

A configuration can be base to the simulator through for example:
```bash
python main.py -m 'WCAE' -c 'sample.WCAE_sample_config'
```

The number of parallel processes is adjuster through '-n xy'. Note that it defines how many models that are trained in parallel, it does not parallelize the model itself, this can be done in the configuration file if supported.

```bash
python main.py -m 'WCAE' -c 'sample.WCAE_sample_config' -n 2
```