# Finetuning Falcon-40b

## Procedure

- Download the weights.
- Prepare the dataset.
- Perform finetuning.

## Setup

1. Clone the Lightning repo

```python
git clone <https://github.com/Lightning-AI/lit-parrot
cd lit-parrot
```

2.Memory Efficiency set-up

```python
# for cuda; gpu_device
pip install --index-url <https://download.pytorch.org/whl/nightly/cu118> --pre 'torch>=2.1.0dev'

# for cpu
pip install --index-url <https://download.pytorch.org/whl/nightly/cpu> --pre 'torch>=2.1.0dev'
```

3.Install the requirements

```python
# in the lit-parrot directory
pip install -r requirements.txt
```

4.Model Wieghts

- Download and extract the weights from `tiiuae/falcon-40b`
- format to lit format.

skip the download part since you have the model weights already: `"/workspace/falcon40b/falcon-40b"`
Direct the script to the path of the model

```python
# download the model weights
python scripts/download.py --repo_id tiiuae/falcon-40b

# convert the weights to Lit-Parrot format
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/tiiuae/falcon-40b
```

5.Prepare data

```python
python scripts/prepare_dev_set.py \
    --destination_path data/alpaca \
    --checkpoint_dir /workspace/falcon40b/falcon-40b
```

6.Finetune

```python
python finetune/adapter_v2.py \
    --data_dir data/alpaca  \
    --checkpoint_dir checkpoints/tiiuae/falcon-40b \
    --out_dir out/adapter/alpaca
```
