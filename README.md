# DePOI

Code for Disentangled Graph Debiasing for Next POI Recommendation

## Model Training

To train our model with default hyper-parameters:

```
python -u main.py --device=cuda --dataset=nyc > log/nyc.log 2>&1 &
```