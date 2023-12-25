# SpotTarget

**Paper**: Jing Zhu*, Yuhang Zhou*, Vassilis N. Ioannidis, Shengyi Qian, Wei Ai, Xiang Song, Danai Koutra
Pitfalls in Link Prediction with Graph Neural Networks: Understanding the Impact of Target-link Inclusion & Better Practices.

*Link*:  https://arxiv.org/abs/2306.00899



**Citation (bibtex)**:
```
@inproceedings{caper,
  title={Pitfalls in Link Prediction with Graph Neural Networks: Understanding the Impact of Target-link Inclusion & Better Practices.},
  author={Jing Zhu*, Yuhang Zhou*, Vassilis N. Ioannidis, Shengyi Qian, Wei Ai, Xiang Song, Danai Koutra},
  booktitle={WSDM},
  year={2024}
}
```
## Training-time Usage
For use as an excluder within the DGL library, check this merged pull requests: https://github.com/dmlc/dgl/pull/5893. 

Example usages are as follows:

```
low_degree_excluder = dgl.dataloading.SpotTarget(
        g,
        exclude="reverse_id",
        degree_threshold=degree_threshold,
        reverse_eids=reverse_eids,
    )
sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude=low_degree_excluder,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
    )
```

For using it separately, check [RFC_low_degree_sampler.md](./RFC_low_degree_sampler.md)

## Test-time Usage

```
python3 leakage_check.py
```

# Question & troubleshooting

If you encounter any problems running the code, pls feel free to contact Jing Zhu (jingzhuu@umich.edu)