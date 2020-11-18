# Simulator for (Topology-Aware) Representation Learning Methods

**IMPORTANT: Work in progress.**

This repository provides a simulation framework for different representation learning algorithms such as Witness Autoencoder, Topological Autoencoder, Connectivity optimized representation learning, UMAP and t-SNE. It is easily extensible, and allows for reproducible and efficient experimenting with the different algorithms on different datasets.


## Installation 

### pre-commit
Pre-commit is used to protect the master branch from (accidental) commits. 

```bash
pip install pre-commit
pre-commit --version
pre-commit install
```
### Requirements
```bash
pip3 install -r requirements.txt
```

### torchph (only needed for COREL)
https://github.com/c-hofer/torchph

Recommended to install it this way: (not with conda)
```bash
git clone https://github.com/c-hofer/torchph.git
cd torchph
python setup.py develop
```



## Usage

The simulator can be used with:
```bash
python main.py
```

One can choose a model, the number of parallel processes and a configuration for the model.
Please refer to scripts/config_library/sample.py, where sample configurations can be found.
Currently the simulator supports:
- Witness Autoencoder (command: -m 'WCAE')
- Topological Autoencoder (TopoAE)[[1]](#1). (command: -m 'topoae')
- t-SNE, UMAP (command: -m 'competitor')
- "COREL"[[2]](#2) (not integrated, needs to be used manually at this point)

For the competitors, the model is defined directly in the configuration file.

A configuration can be passed to the simulator with '-c #name', e.g.
```bash
python main.py -m 'WCAE' -c 'sample.WCAE_sample_config'
```

The simulator has "access" to new configurations stored in scripts/config_library. 

The number of parallel processes can be adjusted by '-n xy', e.g.

```bash
python main.py -m 'WCAE' -c 'sample.WCAE_sample_config' -n 2
```

Note that it defines how many models that are trained in parallel, it does not parallelize the model itself, this can be done in the configuration file if supported.




## References
<a id="1">[1]</a> 
M. Moor, M. Horn, B. Rieck, and K. Borgwardt. Topological Autoencoders. ICML 2020.

<a id="2">[2]</a> 
C. Hofer, R. Kwitt, M. Dixit, and M. Niethammer.
Connectivity-Optimized Representation Learning via Persistent Homology. ICML 2019.

