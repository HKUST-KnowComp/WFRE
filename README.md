# Wasseretein-Fisher-Rao Embedding

The implementation for ACL 2023 findings paper:

> Wasserstein-Fisher-Rao Embedding: Logical Query Embeddings with Local Comparison and Global Transport

by Zihao Wang, Weizhi Fei, Hang Yin, Yangqiu Song, Ginny Y. Wong, and Simon See.

# Prepare the data

The KG data (FB15k, FB15k-237, NELL995) should be put into under 'data/' folder. We use the [data](http://snap.stanford.edu/betae/KG_data.zip) provided in the [KGReasoning](https://github.com/snap-stanford/KGReasoning).
The structure of the data folder should be at least and we follow the query type of EFO-1 and transfer the data.
```
data
	|---FB15k-237-betae
	|---FB15k-betae
	|---NELL-betae
```

The OpsTree is generated by `binary_formula_iterator` in `fol/foq_v2.py`. The overall process is managed in `formula_generation.py`. Transform beta queries to EFO-1 is the next step.

To generate the formula and transform the queries data, just run
```bash
python formula_generation.py
python transform_beta_data.py
```


# Run experiments on different knowledge graphs
The hyperparameters in three datasets is provided in config/papers.
To get our results in paper just run
```bash
python main.py -config config/papers/NELL.yaml
```
