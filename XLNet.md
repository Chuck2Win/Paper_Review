# XLNet : Generalized Autoregressive Pretraining for Language Understanding

## 1. Point  
2 stream self attention in XLNet, Permutation Language Model  

Language Model은 크게 AR model, AE model이 있음 ( AR model은 XLNet, GPT 등, AE model은 Bert )  

* AR Model
** Forward, backward를 동시에 고려하지 못한다.

* AE Model : Reconstruct original data from corrupted input(<-[MASK]가 있는 경우)
** 동시에 고려 가능, 그러나 $$P([MASK_i,MASK_l]|unmasked token)=P([MASK_i]|unmasked token) * P([MASK_l]|unmasked token)$$ 즉 독립 가정, pretraining과 inference(finetunning)시 discrepancy 문제

XLNet은 AR Model과 AE Model의 장점을 합친 버전임.
bidirectional 가능, discrepancy 해결, 독립성 해결

## 2. Permutation Language Model

ex) x1, x2, x3, x4에서 permutation의 경우의수 4!(-> 이것을 다 활용하는 것이 아니라, random sampling 함)  
이 때 x3를 위주로 살펴본다면
 x3 . . . 
 . x3 . .
. . x3 .
. . . x3

| x3 | # | # | # |  
| # | x3 | # | # |  
| # | # | x3 | # |  
| # | # | # | x3 |  

즉 permutation을 하게 되면, x3에 대한 양방향의 정보가 다 들어오게 된다. 그래서 bidirectional 문제 해결 완료
그러나 permutation도 문제점이 존재하는데,
![](https://github.com/Chuck2Win/Paper_Review/blob/master/image/CodeCogsEqn%20(1).gif)
z:factorization order

t-1 까지의 token들로 t 번째 token을 예측함.
예를 들어 설명하면, x4,x2,x3,x1의 것을 sampling했다고 하면, ![$\p(x_z3|x_z1,x_z2)$](https://github.com/Chuck2Win/Paper_Review/blob/master/image/CodeCogsEqn.gif)

기존 transformer가 못하는 사항
1. t번째 token을 예측하는데에 있어서, model은 단순히 t번째 token의 위치만을 고려해야한다.(content은 알면 안된다)
2. 그리고 t번째 token을 예측할 때에 model은 t번째 토큰 이전의 모든 content를 encode 해야한다.
다시 예를 들어서 설명하면, | x1 | x2 | x3 | x4 | 라 할 때에, x3을 예측하기 위해서는 x3의 내용을 모르고 위치만을 고려해야 하며, x3을 예측할 때에는 x1,x2의 content를 encode해야한다는 소리.

이에 대해서 XLNet은 x_t를 embedding 하기 위해서,
Content Representation for x_z(t) : content information from x_z(1)~x_z(t) | Position from x_z(1) to x_z(t)
<- Content stream attention
Query Representation for x_z(t) : content information from x_z(1)~x_z(t-1) | Position x_z(t) only
<- Query stream attention 

example 
input sequence의 order | x1 | x2 | x3 | x4 |
permutation order | x3 | x2 | x4 | x1 |

- Content stream attention에서의 mask
| | x1 | x2 | x3 | x4 |
| x1 | | | | |
| x2 | x |  | | x |
| x3 | x | x | | x |
| x4 | x | | | |

- Query stream attention에서의 mask : 자기 자신의 위치와 이전까지의 token의 내용
| | x1 | x2 | x3 | x4 |
| x1 | x | | | |
| x2 | x | x  | | x |
| x3 | x | x | x | x |
| x4 | x | | | x |
? 그러면 positional encoding할 때에 자기 자신의 위치만 살려서 더해주고, 나머지 embedding(token embedding)만 mask를 씌우나?


