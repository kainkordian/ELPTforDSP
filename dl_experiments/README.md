# Steps

Install environment:

    conda env create -f environment.yaml
  
Switch to environment:

    source activate kordian-loadprediction
  
Start script, e.g.:

    python run.py -dp ../data/avazu_15min.csv -dn avazu -dtc messages -m CNN -d cpu