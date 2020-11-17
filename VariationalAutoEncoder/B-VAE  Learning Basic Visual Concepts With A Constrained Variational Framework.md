# B-VAE : Learning Basic Visual Concepts With A Constrained Variational Framework

## 2017, ICLR, Irina Higgins et. al

cf. **disentangled representation ** can be defined as one where single latent units are sensitive to changes in single generative factors, while being relatively invariant to changes in other factors (Bengio et al., 2013)

 

간단하게 VAE에서 ELBO Term의 KL 부분 앞에 B를 도입함



Contribution 

1) we proposed B-VAE, a new unsupervised approach for learning disentangled representations of independent visual factors

2) we devise a protocol to quantitatively compare the degree of disentanglement learnt by different models

3) 우리의 것이 sota !



## Model

![image-20201117153637155](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117153637155.png)

![image-20201117153835857](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117153835857.png)

아래에서 식 2를 보게 된다면, 쉽게 이야기해서 ELBO Term에서 KL divergence가 $ e $ 미만이 되길 원한다는 식으로 생각.

이 때 KL divergence Term은 approximate posterior distribution이 prior (isotropic unit Gaussian distribution)을 따르고 이 말은 latent variable이 generative factors v를 disentangled manner로 capture하기를 윈한다는 것이다. (**쉽게 말해서 독립성을 따르는 ground truth generative factor, w를 latent variable z가 capture 하게끔 하기 위해서 KL divergence term을 둔다는 것임.**)

![image-20201117153343543](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117153343543.png)

cf. KKT condition

![image-20201117155814464](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117155814464.png)

![image-20201117155909021](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117155909021.png)

KKT condition을 만족한다고 했으니, 충분 조건으로 생각하면 된다.

**We postulate that in order to learn disentangled representations of the conditionally independent data generative factors v, it is important to set β > 1, thus putting a stronger constraint on the latent bottleneck than in the original VAE formulation of Kingma & Welling (2014). **

즉, conditionally independent generative factor를 담아내기 위해서 $\beta$ 를 1보다 크게 하는 것임.

we hypothesis that higher values of $\beta$ should encourage learning a disentangled representation of v.

즉, $\beta$를 크게 할수록 disentangled representation을 더 잘 학습한다고 가정함.

그러나 적절한 $\beta$를 찾는 것이 쉽지는 않다고 한다. (추가적으로 공부할 부분)

## Disentanglement Metric

As stated above, we assume that the data is generated by a ground truth simulation process which uses a number of data generative factors, some of which are **conditionally independent**, and we also assume that they are **interpretable**.

PCA, ICA 등 역시도 representation leaning을 하게 되면 conditionally independent한 variables를 만들 수 있음. 그러나 그러한 variables은 intepretable하다고 할 수 없다.

그래서 우리가 제안하는 것은, intepretablility & independence를 measure하는 것임.

절차를 살펴보자 

이 때 y(position x, position y, scale, rotation) 은 v(independent generative factors)라고 하자.

![image-20201117162401039](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117162401039.png)

![image-20201117162944610](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117162944610.png)

![image-20201117162916532](C:\Users\UNIMAX\AppData\Roaming\Typora\typora-user-images\image-20201117162916532.png)

식 5에서 $ [v_{2,l}]_{k} = [v_{1,l}]_{k} $ if $ k=y $

즉 conditionally independent ground truth generative factors의 차원은 K인데 여기서 하나를 고정시킨다(예시에선 scale)

그런 다음에 x를 생성해내고, (z = v,w 라고 생각하면 될 듯)

이 다음에 x를 활용해서 z를 생성해낸다. (z = mu(x) 로 상정함 -- 이 부분이 나중에 다른 논문에서 문제점으로 제기됨)

그 다음에 서로 빼면 예상으론 scale에 관계 된 z의 feature는 0이 될 터이고. 이것을 단순하게 classifier만 먹여도 p(y|z_diff)는 쉽게 scale에 관계된 부분이라고 예측할 수 있겠지 (y=[0,0,1,0]으로 하면 되지 않을 까 함.)

## Conclusion

Unlike InfoGAN and DC-IGN. our approach does not depend on any a priori knowledge about the number or the nature of data generative factors 

<- 근데 metric을 활용하려면 개수를 알고 있어야 될 것 같네 ( 이것도 추후에 문제가 되겠군 )

< - 즉 정말로 unsupervised learning이냐 이거지.. inductive bias가 들어갈텐데..



## 나의 정리

- Disentangled representation learning 임 (처음 접함)
- VAE에서 KL Term에 $\beta$를 도입해서 매우 간단하게 해서 representation learning 실시
- 측정 방식 (참신했음.)
- 허나, 정말로 unsupervised 하냐에 대해서는 물음표 