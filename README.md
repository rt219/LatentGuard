# Latent Guard
[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://latentguard.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-URL-green.svg)](https://raw.githubusercontent.com/rt219/LatentGuard/main/dataset/CoPro_v1.0.json)
[![Poster](https://img.shields.io/badge/Poster-PDF-blue.svg)](https://latentguard.github.io/poster_v4.pdf)
[![Video](https://img.shields.io/badge/Video-Youtube-ff69b4.svg)](https://www.youtube.com/watch?v=MLZxFOL1NAE)
[![中文Blog](https://img.shields.io/badge/中文-博客Blog-darkblue.svg)](https://zhuanlan.zhihu.com/p/794411466)

<!-- *Democratic AI Safety: No one can define safety for all — Latent Guard helps you define yours.* -->
<img width="766" alt="image" src="https://github.com/user-attachments/assets/c6bd9e2f-4e38-4212-b0fb-db8d9daf67bc">

---

An open-source, efficient and extensible framework for enhancing safety🛡️ in text-to-image (T2I) generation🖼️, designed to prevent misuse and improve flexibility. It is built to be accessible, speedy, and reliable for the entire community. 
- **⚡️Fast**: Detects unsafe input prompts in approximately **1ms** and can be trained on a single NVIDIA 3090 GPU.
- **🔧Extensible**: Supports customized unsafe concepts to block; compatible with all T2I models based on text encoders, SD/SDXL etc.
- **🏆State-of-the-art**: Faster performance, higher accuracy, and better scalability than existing safety methods.
- **🌐Open**: All processes—data generation, training, testing, and inference—are fully open-source.

📰 _Latent Guard_ has been reported by famous tech media outlets like [TechXplore](https://techxplore.com/news/2024-04-framework-safety-text-image-generation.html) and [MarkTechPost](https://www.marktechpost.com/2024/05/03/latent-guard-a-machine-learning-framework-designed-to-improve-the-safety-of-text-to-image-t2i-generative-networks/).

## Latest News
**[2024/09/25 New]🚀🚀🚀:** We released our **code**📝 and the **model weights**⚙️!

**[2024/07/15]:** We released our dataset **CoPro** in *dataset/CoPro_v1.0.json*.

**[2024/07]:** Our paper has been accepted by ECCV 2024.

This is the official repo of the paper accepted by ECCV 2024 [Latent Guard: a Safety Framework for Text-to-image Generation(arXiv)](https://arxiv.org/abs/2404.08031).

    @article{liu2024latent,
      title={Latent Guard: a Safety Framework for Text-to-image Generation},
      author={Liu, Runtao and Khakzar, Ashkan and Gu, Jindong and Chen, Qifeng and Torr, Philip and Pizzati, Fabio},
      journal={arXiv preprint arXiv:2404.08031},
      year={2024}
    }

# Dataset _CoPro_ and model weights

The dataset is in the repository **CoPro** in **`dataset/CoPro_v1.0.json`** and the model weights are stored in **`model_parameters.pth`**.

# Environment
To set up the conda environment, run the following command (this process takes around 10 minutes depending on the network and server):
```bash
conda env create -f latentguard.yml
```
After installation, activate the environment with:
```bash
conda activate latentguard
```

# Inference

To run the inference, execute the following command:

```bash
python inference.py
```

Or you can choose to specify the following parameters according to your requirements:

```bash
python inference.py --file_path FILEPATH --threshold VALUE
```

- `--file_path`: Specifies the path to the unsafe file. The default value is `'unsafe_sample.txt'`(a file in this repo), which provides a sample format for the prompt data to be detected.
- `--threshold`: This parameter determines the threshold value of predicting unsafe or not. It can be adjusted based on the data distribution.

# Test and Train on our dataset CoPro
## Preprocessing clip_cache.pt file
To improve the speed of testing and training, it's necessary to preprocess the `clip_cache.pt` file, which stores the CLIP embedding representations of the prompts. This process may take over 20 minutes and will display a progress bar.

Run `python prepare.py` to obtain the `clip_cache.pt` file. 

Once the execution is complete, the `clip_cache.pt` file will be generated. The path has already been set in `config.py`, so **no** manual configuration is needed. You can proceed with the subsequent commands.

## Testing
You can simply run `python test.py` to obtain the results for **Table 1b**.

## Training
You can simply run `python main.py` to train _Latent Guard_ on _CoPro_. 

Our model takes **only** **30** **minutes**⚡️ to train on a single NVIDIA 3090 GPU.

# Motivation & Background
<p align="center">
  <img width="774" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/77c982d7-c8b4-4961-91b8-4264e7fc33b1">
</p>

Recent text-to-image generators are composed of a text encoder and a diffusion model. Their deployment without appropriate safety measures creates risks of misuse (left). We propose _Latent Guard_ (right), a safety method designed to block malicious input prompts. Our idea is to detect the presence of blacklisted concepts on a learned latent space on top of the text encoder. This allows to detect blacklisted concepts beyond their exact wording, extending to some adversarial attacks too ("**\<ADV\>**"). The blacklist is adaptable at test time, for adding or removing concepts without retraining. Blocked prompts are not processed by the diffusion model, saving computational costs.

# Abstract
With the ability to generate high-quality images, text-to-image (T2I) models can be exploited for creating inappropriate content. To prevent misuse, existing safety measures are either based on text blacklists, which can be easily circumvented, or harmful content classification, requiring large datasets for training and offering low flexibility. Hence, we propose _Latent Guard_, a framework designed to improve safety measures in text-to-image generation. Inspired by blacklist-based approaches, _Latent Guard_ learns a latent space on top of the T2I model's text encoder, where it is possible to check the presence of harmful concepts in the input text embeddings. Our proposed framework is composed of a data generation pipeline specific to the task using large language models, ad-hoc architectural components, and a contrastive learning strategy to benefit from the generated data. The effectiveness of our method is verified on three datasets and against four baselines. 

# Approach 
<p align="center">
<img width="782" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/4650feb8-63d6-4d35-9a21-88365406a9d1">
</p>

**Overview of _Latent Guard_.** We first generate a dataset of safe and unsafe prompts centered around blacklisted concepts (left). Then, we leverage pretrained textual encoders to extract features, and map them to a learned latent space with our Embedding Mapping Layer (center). Only the Embedding Mapping Layer is trained, while all other parameters are kept frozen. We train by imposing a contrastive loss on the extracted embedding, bringing closer the embeddings of unsafe prompts and concepts, while separating them from safe ones (right).

# Dataset **_CoPro_** Generation
<p align="center">
<img width="1099" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/f27cad9d-e078-4763-8f7d-85724753d6c0">
</p>

**_CoPro_ generation.** For $\mathcal{C}$ concepts, we sample unsafe $\mathcal{U}$ prompts with an LLM as described in Section 3.1. Then, we create Synonym prompts by replacing $c$ with a synonym, also using an LLM, and obtaining $\mathcal{U}^\text{syn}$. Furthermore, we use an adversarial attack method to replace $c$ with an "**\<ADV\>**" Adversarial text ($\mathcal{U}^\text{adv}$). Safe prompts $\mathcal{S}$ are obtained from $\mathcal{U}$. This is done for each ID and OOD data.

# Qualitative and Quantitative Results
**Evaluation on _CoPro_.**
We provide accuracy (a) and AUC (b) for _Latent Guard_ and baselines on _CoPro_. We either rank first or second in all setups, training **only** on Explicit ID training data. We show examples of prompts of _CoPro_ and generated images in (c). The unsafe image generated advocate the quality of our dataset. _Latent Guard_ is the only method blocking all the tested prompts.

<p align="center">
<img width="1063" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/d5b95664-b160-4da6-8352-48e83f8d9931">
</p>

**Evaluation on Unseen Datasets**
We test *Latent Guard* on existing datasets for both Unsafe Diffusion and I2P++. Although the input T2I prompts distribution is different from the one in _CoPro_, we still outperform all baselines and achieve a robust classification.

<p align="center">
<img width="911" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/16ec46a3-db33-4d06-955b-ce4026c9d4aa">
</p>

# Speed and Feature Space Analysis
<p align="center">
<img width="454" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/8c38a33b-41b1-475f-a36c-c43138865025">
</p>

**Computational cost.** We measure processing times and memory usage for different batch sizes and concepts in $\mathcal{C}_\text{check}$. In all cases, requirements are limited.

<p align="center">
<img width="547" alt="image" src="https://github.com/rt219/LatentGuard/assets/45531420/20f1a0ac-cae6-48ae-82dc-06dabeff33f1">
</p>

**Feature space analysis.** Training _Latent Guard_ on _CoPro_ makes safe/unsafe regions naturally emerge (right). In the CLIP latent space, safe/unsafe embeddings are mixed (left).
