
# scMulti

scMulti is a multitask transfer learning model for integrating paired, unpaired and mosaic single-cell multi-omics data. scMulti leverages hypergraph neural network to capature global graph-level gene and peak information, learns a low-dimensional embedding for each modality, devises a dual-attention aggregation mechanism to dynamically asses the significance of various modalities.

## Installation

scMulti can be obtained by simply clonning the github repository:

```

git clone https://github.com/kanyulongkkk/scMulti.git
```

The following python packages are required to be installed before running scJoint:
`h5py`, `torch`, `itertools`, `scipy`, `numpy`,  `os`, `random`, `sys`, `time`, and `datetime`.


In terminal, run

```
python main.py
```

The output will be saved in `./output` folder.


# scMulti
Atlas-level data integration in Single cell genomics with Multitask Hypergraph Attention Mechanism
