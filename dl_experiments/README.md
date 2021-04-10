# Steps

Install environment:

    conda env create -f environment.yaml
  
Switch to environment:

    source activate kordian-loadprediction
  
Start script, e.g.:

    python run.py -drd ../data -dn avazu -dsr 15min -dtc messages -m GRU -d cpu -cr 1 -gr 0
