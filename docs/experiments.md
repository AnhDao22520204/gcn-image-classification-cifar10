# Experiments

## Reported Patch-GCN training setup

- Loss: Cross Entropy Loss
- Optimizer: AdamW
- Learning rate: `1e-3`
- Batch size: `128`
- GCN layers: `3`
- MLP head: `512 → 512 → 10`
- Epochs: `30`

## Reported results

| Model | Test Loss | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Patch-GCN | 1.2137 | 0.5679 | 0.5658 | 0.5679 | 0.5621 |
| CNN-GCN | 0.2651 | 0.9206 | 0.9204 | 0.9206 | 0.9204 |

## k optimization

The report compares `k = 2, 4, 6` and states that `k = 2` produced the best result.
