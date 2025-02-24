# This is the dev branch for the NPF presentation!

# The Neural Process Family: Demo Code Repository

Welcome to the demo repository for our presentation on **The Neural Process Family: Survey, Applications and Perspectives**. This repository contains notebook implementations of three key Neural Process variants:

- **Conditional Neural Processes (CNPs)**
- **Neural Processes (NPs)**
- **Attentive Neural Processes (ANPs)**

These implementations are adapted from original work by DeepMind and enhanced with our own insights to support our survey paper and presentation.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Code Overview](#code-overview)
3. [Mathematical Explanation](#mathematical-explanation)
   - [Conditional Neural Process (CNP)](#conditional-neural-process-cnp)
   - [Attentive Neural Process (ANP)](#attentive-neural-process-anp)
4. [Installation](#installation)
5. [References and Citations](#references-and-citations)
6. [Contact and Acknowledgments](#contact-and-acknowledgments)
7. [Disclaimer](#disclaimer)

---

## Introduction

Neural Processes (NPs) have emerged as a powerful family of models that blend the flexibility of neural networks with the uncertainty estimation of Bayesian methods. They learn a mapping from a set of observed data points to a distribution over functions, which enables robust predictions with well-calibrated uncertainty estimates.

This repository accompanies our presentation:
> **The Neural Process Family: Survey, Applications and Perspectives**  
> *Paper Authors*: Saurav Jha, Dong Gong, Xuesong Wang, Richard E. Turner, Lina Yao  
> *Presenters*: Alex Inch, Eric Ma, Skye Purchase, and Yuan Lu  

We provide demo code for three major NP variants:
1. **Conditional Neural Processes (CNPs)**
2. **Neural Processes (NPs)**
3. **Attentive Neural Processes (ANPs)**

For detailed background, please refer to our survey paper and the original publications.

---

## Code Overview

- **`conditional_neural_process.ipynb`**:  
  Implements the **Conditional Neural Process (CNP)**. This notebook explains the core building blocks of the model, such as the encoder, the aggregation (typically via mean pooling), and the decoder.

- **`attentive_neural_process.ipynb`**:  
  Implements both the **Neural Process (NP)** and the **Attentive Neural Process (ANP)**. Here, the inclusion of attention mechanisms in ANPs is demonstrated to enhance the model's ability to capture local variations in the data.

Each notebook covers:
- Model architecture details
- Training and evaluation loops
- Visualizations comparing predictions with ground truth

---

## Mathematical Explanation

### Conditional Neural Process (CNP)

Given a context dataset:

$$C = \{(x_i, y_i)\}_{i=1}^{N}$$

the CNP operates as follows:

1. **Encoding**: Each context pair ${x_i, y_i}$ is passed through an encoder function to produce a representation:

$$r_i = h(x_i, y_i)$$

2. **Aggregation**: The individual representations are combined into a single global representation. A common choice is to take the mean:

$$r = \frac{1}{N} \sum_{i=1}^{N} r_i$$

3. **Decoding**: For each target input $x_t$$ in the target set 

$$T = \{x_t\}_{t=1}^{M},$$

a decoder produces the predictive distribution:
$$p(y_t \mid x_t, r) = \mathcal{N}(\mu(x_t, r), \sigma(x_t, r))$$
where $\mu$ and $\sigma$ are outputs of the decoder network.

This formulation allows the CNP to capture uncertainty in its predictions while maintaining computational efficiency.

### Attentive Neural Process (ANP)

The Attentive Neural Process enhances the CNP by introducing a target-specific attention mechanism. Instead of a single aggregated representation $r$, each target point $x_t$ receives its own context representation through attention.

1. **Encoding**: As in the CNP, each context pair is encoded:

$$r_i = h(x_i, y_i)$$

2. **Attention Mechanism**: For a given target $x_t$, compute attention weights over the context points:

$$\alpha(x_t, x_i) = \frac{\exp(e(x_t, x_i))}{\sum_{j=1}^{N} \exp(e(x_t, x_j))}$$

where $e(x_t, x_i)$ is a learned compatibility function, often implemented as a small neural network.

3. **Target-Specific Aggregation**: The target-specific representation is computed as a weighted sum:

$$r_t = \sum_{i=1}^{N} \alpha(x_t, x_i) \, r_i$$

4. **Decoding**: The decoder now uses the target-specific representation $r_t$ to produce the predictive distribution:

$$p(y_t \mid x_t, r_t) = \mathcal{N}(\mu(x_t, r_t), \sigma(x_t, r_t))$$

By incorporating attention, the ANP can focus on the most relevant context points for each target, improving predictive accuracy, especially in settings with non-uniform data distributions.

---

## Installation

**Prerequisite**: Python 3.7 (or above)

### Option 1: Google Colab

The fastest way to explore the demo is via [Google Colab](https://colab.research.google.com):

- **CNP Notebook**: [Conditional Neural Processes](https://colab.sandbox.google.com/github/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb)
- **(A)NP Notebook**: [Attentive Neural Processes](https://colab.sandbox.google.com/github/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb)

Colab provides a pre-configured environment with TensorFlow (1.13.1), NumPy (1.14.6), and Matplotlib (2.2.4).

### Option 2: Local Environment

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo
2. Create and activate a virtual environment:
   ```bash
   python3.7 -m venv env
   source env/bin/activate
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
5. Open the provided '.ipynb' files to run the demos.

## References and Citations

If you find this repository useful in your research or presentations, here are some work we are citing:
1. **The Neural Process Family: Survey, Applications and Perspectives** Saurav Jha, Dong Gong, Xuesong Wang, Richard E. Turner, Lina Yao. 2022.

1. **Conditional Neural Processes**: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D, Shanahan M, Teh YW,
Rezende DJ, Eslami SM. *Conditional Neural Processes*. In International Conference
on Machine Learning 2018.

1. **Neural Processes**: Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D.J., Eslami, S.M. and Teh, Y.W. *Neural processes*. ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models 2018.

1. **Attentive Neural Processes**: Kim, H., Mnih, A., Schwarz, J., Garnelo, M., Eslami, A., Rosenbaum, D., Vinyals, O. and Teh, Y.W. *Attentive Neural Processes*. In International Conference on Learning Representations 2019.

## Contact and Acknowledgments

**Presentation Team** (Equal Contribution)
- Alex Inch
- Eric Ma
- Skye Purchase
- Yuan Lu

**Paper Authors**
- Saurav Jha
- Dong Gong
- Xuesong Wang
- Richard E. Turner
- Lina Yao

For questions regarding the presentation or the survey paper, please feel free to contact any of the presenters. Your feedback is highly appreciated!

## Disclaimer

This repository contains demonstration code and is not an official product. Use this code at your own risk.

Enjoy exploring the Neural Process Family and happy coding!
