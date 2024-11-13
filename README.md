## About

This repository implements the methods used in the paper "Logistic Variational Bayes Revisited" (2024)

TLDR: We introduce SOTA methods for variational logistic regression and GP classification

https://arxiv.org/abs/2406.00713

## Installing environment

The following command will install the environment for the project.

```bash
conda env create -f environment.yml
```


## Quick start GP Classification

If you want to do GP classfication see

https://github.com/mkomod/vi-per/blob/main/notebooks/gp_simulations.ipynb

```python
from src._97_gpytorch import LogisticGPVI

model = LogisticGPVI(y, X, n_inducing=50, n_iter=200, verbose=False)
model.fit()

y_pred = model.predict(X)
```

If you are familiar with Gpytorch the following class is an implementation of VI-PER.

```python
class LogitLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    has_analytic_marginal = False

    def __init__(self, l_max=12.0, ):
        self.l_max = l_max
        self.l = torch.arange(1.0, self.l_max*2, 1.0, requires_grad=False)
        return super().__init__()
 
    def forward(self, function_samples, *args, **kwargs):
        """ defines the liklihood function """
        output_probs = torch.sigmoid(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)

    @torch.jit.export
    def expected_log_prob(self, y, function_dist, *args, **kwargs):
        """ compute the expected log probability """
        M = function_dist.mean.view(-1, 1)
        S = function_dist.stddev.view(-1, 1)
        V = S**2

        M_S = M / S
        ML = M * self.l
        SL = S * self.l
        VL = 0.5 * V * (self.l ** 2)
        
        y_M = torch.dot(y, M.squeeze())
        normal_term = torch.sum(S / math.sqrt(2 * torch.pi) * torch.exp(-0.5 * M**2 / V) + M * ndtr(M_S))
        series_term = torch.sum(
            (-1.0)**(self.l - 1.0) / self.l * (
                torch.exp(ML + VL + log_ndtr(-M_S - SL)) + torch.exp(-ML + VL + log_ndtr(M_S - SL))
            )
        )

        return y_M - normal_term - series_term
```



## Project structure

The project is structured as follows:

```
.
├── README.md
├── data (NOT INCLUDED IN REPOSITORY AS NOT PUBLICLY AVAILABLE YET) 
├── environment.yml
├── notebooks
├── results
├── figures
├── scripts
└── src
    ├── __00_funcs.py
    ├── __01__data_generation.py
    ├── __02__method.py
    ├── __03__simulations.py
    ├── __04__application.py
    ├── __05__figures.py
    ├── __06__tables.py
    ├── __07__gp_simulations.py
    ├── __08__earthquake.py
    └── __97__gpytorch.py
```

The `data` folder contains the data used in the project. It is not included in the repository as it is not publicly available yet.

The `notebooks` folder contains the notebooks used for exploratory data analysis and for the generation of the figures for GP simulations and the application to soil liquefaction.

The `results` folder contains the results of the simulations and the applications.

The `scripts` folder contains the scripts used for the generation of the results.

The `src` folder contains the source code of the project.


## Reproducing results

The results for the following sections can be reproduced by running the scripts in the `scripts` folder:

- Section 3.1: `01-logistic_regression_simulations.sh`
- Section 3.2: `02-gaussian_process_example.sh`
- Section 4.1: `03-earthquake.sh`
- Section 4.2: `04-applications.sh`

The results will be saved to the `results` folder.

## Generating figures

The figures can be reproduced by running the `__05__figures.py` script in the `src` folder. Furthermore, the figures for the GP example can be reproduced by running the gp_simulations notebook in the `notebooks` folder. The figures for the earthquake application can be reproduced by running the soil_liquefaction notebook in the `notebooks` folder. The figures will be saved to the `figures` folder.

## Visdom
to visualize on web, you have to start visdom server. Result is on localhost:8900
1. create screen: 
   screen -S visdom.8900
2. start visdom server:
   python -m visdom.server -p 8900
3. leave screen: 
   ctrl + a + d

## Backbone and Celeba
It is heavily based on https://github.com/pangwong/pytorch-multi-label-classifier/tree/master
Checkout data dir on that github to proceed with data

## Examples of commands:

- If want to see visdom remember to initialize appropriate port + Activate conda.

- Point-wise softmax
```
nohup python src/combine.py --dir "/shared/sets/datasets/vision/artificial_shapes/size64_onlydisk_simplicity2_len10240_cbF_cfT_noF" --rez_dir "/shared/results/pyla/multilabel/arti/" --mode "Train" --model "Lenet" --name "lenet_64_notsq_cpu" --batch_size 64 --gpu_ids -1 --input_channel 3 --load_size 64 --input_size 64 --mean [0,0,0] --std [1,1,1] --ratio "[0.94, 0.03, 0.03]" --shuffle --load_thread 8 --sum_epoch 30 --lr 0.001 --lr_mult_w 1.0 --lr_mult_b 1.0 --lr_decay_in_epoch 1 --display_port 8903 --validate_ratio 1.0 --top_k "(1,)" --score_thres 0.1 --html --display_train_freq 20 --display_validate_freq 20 --save_epoch_freq 1  --display_image_ratio 0.2 > out3.sh &
```

- Bayesian squeezed
```
python src/combine_bayesian.py --dir /shared/sets/datasets/vision/artificial_shapes/size64_simplicity3_len10240_cbF_cfT_noF --rez_dir /shared/results/pyla/multilabel/arti --mode Train --model Lenet --binary_squeezed --name BayesSqueeze64Lenet --batch_size 1000 --gpu_ids 0 --input_channel 3 --load_size 64 --input_size 64 --mean \[0,0,0\] --std \[1,1,1\] --ratio \[0.94,\ 0.03,\ 0.03\] --shuffle --load_thread 8 --sum_epoch 50 --lr 0.0001 --weight_decay 0.0001 --lr_mult_w 1.0 --lr_mult_b 1.0 --lr_decay_in_epoch 1 --gradient_clipping 100.0 --display_port 8902 --validate_ratio 1.0 --top_k \(1,\) --score_thres 0.5 --display_train_freq 10 --display_validate_freq 10 --save_epoch_freq 1 --display_image_ratio 0.2 --bayesian --method tb --l_max 10
```

- Bayesian squeezed only disk
```
python src/combine_bayesian.py --dir /shared/sets/datasets/vision/artificial_shapes/size64_onlydisk_simplicity3_len10240_cbF_cfT_noF --rez_dir /shared/results/pyla/multilabel/arti --mode Train --model Lenet --binary_squeezed --name BayesSqueeze64LenetOnlyDisk --batch_size 256 --gpu_ids 0 --input_channel 3 --load_size 64 --input_size 64 --mean \[0,0,0\] --std \[1,1,1\] --ratio \[0.9,\ 0.05,\ 0.05\] --shuffle --load_thread 8 --sum_epoch 50 --lr 0.0001 --weight_decay 0.0001 --gamma 1.0 --lr_decay_in_epoch 1 --lr_mult_w 1.0 --lr_mult_b 1.0 --gradient_clipping 100.0 --display_port 8902 --validate_ratio 1.0 --top_k \(1,\) --score_thres 0.5 --display_train_freq 10 --display_validate_freq 10 --save_epoch_freq 1 --display_image_ratio 0.05 --bayesian --method tb --l_max 10
```

- Point-wise synthetic squeezed
```
nohup python src/combine.py --dir /home/pyla/bayesian/vi-per/data/synthetic/dim10_len10240_seed0 --rez_dir /shared/results/pyla/multilabel/synthetic --mode Train --model SimpleMLP --binary_squeezed --notimage_type --hidden_size 128 --name test0sqLikeColab --batch_size 1024 --gpu_ids 0 --input_channel 10 --load_size 1 --input_size 1 --ratio \[0.9,\ 0.05,\ 0.05\] --shuffle --load_thread 8 --loss_weight \[1.0,\ 0.0,\ 0.0,\ 0.0\] --sum_epoch 5 --lr 1.0 --momentum 0.9 --lr_mult_w 1.0 --lr_mult_b 1.0 --gamma 1.0 --lr_decay_in_epoch 1 --display_port 8910 --validate_ratio 1.0 --top_k \(1,\) --score_thres 0.5 --display_train_freq 1 --display_validate_freq 1 --save_epoch_freq 1 --display_image_ratio 0.0 > out1.sh &
```

- Bayesian synthetic squeezed
```
nohup python src/combine_bayesian.py --dir /home/pyla/bayesian/vi-per/data/synthetic/lownoise_dim10_le
n10240_seed1 --rez_dir /shared/results/pyla/multilabel/synthetic --mode Train --model SimpleMLP --binary_squeezed --notimage_type --name Bayesnewdat0synth0test0sqlownoise --batch_size 256 --gpu_ids 0 --hidden_size 128 --input_channel 10 --load_size 1 --input_size 1 --ratio \[0.9,\ 0.05,\ 0.05\] --shuffle --load_thread 8 --sum_epoch 5 --lr 0.00001 --weight_decay 0.0 --lr_decay_in_epoch 1 --gamma 1.0 --lr_mult_w 1.0 --lr_mult_b 1.0 --display_port 8915 --validate_ratio 1.0 --top_k \(1,\) --score_thres 0.5 --display_train_freq 1 --display_validate_freq 1 --save_epoch_freq 1 --display_image_ratio 0.0 --bayesian --method tb --l_max 10 > out2.sh &
```
