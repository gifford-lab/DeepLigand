# DeepLigand

## Environment setup
We provide a conda environment in which all necessary packages have been installed. To create and activate this enviroment:

```
conda env create -f environment.yml
source activate deepligand
python update_bilm.py
# to exit this environment do `source deactivate`
```

## Preprocess
```
python preprocess.py -f $INFILE -o $OUTDIR
```
- `INFILE`: a file of MHC-peptide pair to predict on ([example](https://github.com/gifford-lab/DeepLigand/blob/master/example/test))
- `OUTDIR`: output directory

## Predict

```
python main.py -p $OUTDIR/test.h5.batch -o $OUTDIR/prediction 
```
- `OUTDIR`: output directory
