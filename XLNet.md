# XLNet : Generalized Autoregressive Pretraining for Language Understanding

## 1. Point  
2 stream self attention in XLNet, Permutation Language Model  

Language Model은 크게 AR model, AE model이 있음 ( AR model은 XLNet, GPT 등, AE model은 Bert )  

* AR Model
** Forward, backward를 동시에 고려하지 못한다.

* AE Model : Reconstruct original data from corrupted input(<-[MASK]가 있는 경우)
** 동시에 고려 가능, 그러나 $P([MASK_i,MASK_l]|unmasked token)=P([MASK_i]|unmasked token) * P([MASK_l]|unmasked token)$ 즉 독립 가정, pretraining과 inference(finetunning)시 discrepancy 문제

XLNet은 AR Model과 AE Model의 장점을 합친 버전임.
bidirectional 가능, discrepancy 해결, 독립성 해결

## 2. Permutation Language Model

ex) x1, x2, x3, x4에서 permutation의 경우의수 4!(-> 이것을 다 활용하는 것이 아니라, random sampling 함)  
이 때 x3를 위주로 살펴본다면,


