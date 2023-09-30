# Permeability_prediction
Permeability prediction of complex porous materials by conjugating generative adversarial and convolutional neural networks

# pre-requisites

you should install the package

```python
pip install torch
pip install h5py
pip install tifffile
```

# overview

The GitHub repository has some folders like the document tree shows

```powershell
│  README.md
│
├─CNN
│      CNN.py
│      CNN_test.py
│      dataset_CNN.py
│      models.py
│
├─data
│      berea.tif
│
├─datasets
│      berea64_3D
│      berea64_3Doriginal_porosity.csv
│
├─GAN
│      cPro_train.py
│      dataset.py
│      models.py
│      test.py
│
└─preprocessing
        data_preprocesssing.py
        porosity_anlysis.py
```

## original data

The document `berea.tif` in folder `CNN` is the original CT data, you can also get it from [PorousMediaGan/data at 1.0 · LukasMosser/PorousMediaGan (github.com)](https://github.com/LukasMosser/PorousMediaGan/tree/1.0/data).

## preprocessing

you can use the script `data_preprocesssing.py` to construct sub datasets. The command is 

```bash
python data_preprocesssing.py
```

you can the script `porosity_anlysis.py` to calculate the porosity. The command is 

```
python porosity_anlysis.py
```

## train the GAN network

you can use the script `cPro_train.py` to train the GAN network

## train the CNN network

you can use the script `CNN.py` to train the GAN network
