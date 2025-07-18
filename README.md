<div align="center">
<h2>RoboMP<sup>2</sup>: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models</h2>

[**Qi Lv**](https://aopolin-lv.github.io/)<sup>1,2</sup> · [**Hao Li**](https://openreview.net/profile?id=~Hao_Li59)<sup>1</sup> · [**Xiang Deng**](https://xiang-deng-dl.github.io/)<sup>1&dagger;</sup> · [**Rui Shao**](https://rshaojimmy.github.io/)<sup>1</sup> · [**Michael Yu Wang**](https://scholar.google.com/citations?user=Oo7c22wAAAAJ&hl=zh-CN)<sup>2</sup> · [**Liqiang Nie**](https://liqiangnie.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>Harbin Institute of Technology (Shenzhen)&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Great Bay University&emsp;&emsp;&emsp;&emsp;

<sup>&dagger;</sup>corresponding author

**ICML 2024**

<a href="https://arxiv.org/abs/2404.04929"><img src='https://img.shields.io/badge/arXiv-RoboMP2-red' alt='Paper PDF'></a>
<a href='https://aopolin-lv.github.io/RoboMP2.github.io/'><img src='https://img.shields.io/badge/Project_Page-RoboMP2-green' alt='Project Page'></a>
</div>

This work presents RoboMP<sup>2</sup>, a multimodal perception-planning framework with multimodal large language model for manipulation tasks.



## Performance
Here we present the performance comparsion between RoboMP<sup>2</sup> and baseline models.


| Model | L1 | L2 | L3 | L4 | Avg. |
|-------|-----|-----|-----|-----|------|
| **End-to-end models** |
| Gato<sup>&dagger;</sup> | 58.1 | 53.2 | 43.5 | 12.4 | 41.8 |
| Flamingo<sup>&dagger;</sup> | 47.5 | 46.0 | 40.8 | 12.1 | 36.6 |
| VIMA<sup>&dagger;</sup> | 81.6 | 81.5 | 79.0 | 48.9 | 72.7 |
| RT-2 | 72.8 | 70.3 | 66.8 | 47.0 | 64.2 |
| **MLLM Planners** |
| CaP | 71.2 | 70.0 | 42.8 | 44.7 | 57.2 |
| VisualProg | 49.7 | 47.7 | 69.9 | 52.2 | 54.9 |
| I2A<sup>&dagger;</sup> | 77.0 | 76.2 | 66.6 | 49.8 | 65.0 |
| **RoboMP<sup>2</sup> (Ours)** | **89.0** | **85.9** | **86.8** | **68.0** | **82.4** |



## Usage 

### Installation
1. Install the required packages with the provided requirements.txt
```bash
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```

* If you can't install gym==2.21.0,which is necessary for this project, try the following two installation, then the gym will be installed successfully!

  ```python
  pip install setuptools==65.5.0
  pip install --user wheel==0.38.0
  ```

2. Install the VIMABench with [VIMABench](https://github.com/vimalabs/VimaBench).

### Running

1. Change the OpenAI API-key in `data_process/gptutils.py`.

2. Download the [SentenceBert model](https://github.com/UKPLab/sentence-transformers) and change the path of SentenceBert in `retrieval/similarity_retrieval.py`.

3. Put the path of MLLM in `model/custom_model.py`.

4. run the `eval.py`.


## Citation

If you find this project useful, please consider citing:

```bibtex

@InProceedings{pmlr-v235-lv24a,
  title = 	 {{R}obo{MP}$^2$: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models},
  author =       {Lv, Qi and Li, Hao and Deng, Xiang and Shao, Rui and Wang, Michael Y and Nie, Liqiang},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {33558--33574},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/lv24a/lv24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/lv24a.html},
}
```
