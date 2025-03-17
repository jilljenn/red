## Installation of the environment

```bash
conda create --name envir python=3.11.9 -y
conda activate envir

python3 -m pip install -r requirements.txt

python3 -m pip cache purge
conda clean -a -y
```

## Files

- *data.py*: data generation
- *main.py*: script to run the experiments
- *plots.py*: creates figures
- *policies.py*: contains the strategies to recommend a set of items
- *simulate.py*: implements pipelines 
- *tools.py*: useful functions
- *requirements.txt* dependencies
- *README.md*
- *config.yml*: configuration 

## Execute experiments

```bash
python3 -m main
```

Outputs a plot with the corresponding time-dependent curves for the reward ("booking"), the diversity intra batch (scalar products between item embeddings in the batch is higher than a threshold), the diversity inter batch (have the categories in the recommended set of items already been proposed to this user?) and the histogram of average probability of recommendation across users at the end of the horizon.
