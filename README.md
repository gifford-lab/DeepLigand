# DeepLigand

## Data

The 5-fold cross-validation split used in the paper can be downloaded from [here](http://gerv.csail.mit.edu/deepligand_CVdata/). The DeepLigand model provided in this repository is trained on all the five folds combined.


## Environment setup

#### Prerequisites

- R > 3.3
- CUDA 8.0 with cudnn 5.1

#### Conda environment
With the above prerequisites installed, install and activate a [Conda](https://docs.conda.io/en/latest/) environment with all necessary Python packages by:

```
conda env create -f environment.yml
source activate deepligand
python update_bilm.py
```

To deactivate this environment:

```
source deactivate
```

## Preprocess
```
python preprocess.py -f $INFILE -o $OUTDIR
```
- `INFILE`: a file of MHC-peptide pair to predict on ([example](https://github.com/gifford-lab/DeepLigand/blob/master/examples/test)). The names of the MHC supported are listed in the first column of [this](https://github.com/gifford-lab/DeepLigand/blob/master/data/MHC_pseudo.dat) file.
- `OUTDIR`: output directory

## Predict

```
python main.py -p $OUTDIR/test.h5.batch -o $OUTDIR/prediction 
```
- `OUTDIR`: output directory
