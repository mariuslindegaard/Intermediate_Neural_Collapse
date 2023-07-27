### Code for the fixed ETFs experiment

1) Run `python intermediate_nc_etf_mnist.py --num-etf 6 --etf` to train model with 4 trained layers and 6 ETFs
2) Repeat with `--num-etf 2 ... 9`
3) Run `python measure_accuracies_int_etfs_mnist.pyt` to get train and test accuracies for all models
4) plot using `plotting/etf_plotting.ipynb`