# XLNet : Generalized Autoregressive Pretraining for Language Understanding

## 1. Point  
2 stream self attention in XLNet, Permutation Language Model  

Language Model은 크게 AR model, AE model이 있음 ( AR model은 XLNet, GPT 등, AE model은 Bert )  

* AR Model
![AR](https://github.com/Chuck2Win/Paper_Review/blob/master/image/1.png)
** Forward, backward를 동시에 고려하지 못한다. ( multi layer LM이고, bidirectional이라 하더라도 다음 layer에 전달할 때에 그냥 concat하는 수준이므로 )

* AE Model : Reconstruct original data from corrupted input(<-[MASK]가 있는 경우)
![AR](https://github.com/Chuck2Win/Paper_Review/blob/master/image/2.png)
corrupted version : x_hat / masked token : x_bar / m_t=1은 x_t가 masked 된 것을 의미함.
** 동시에 고려 가능, 그러나 P([MASK_i,MASK_l]|unmasked token)=P([MASK_i]|unmasked token) * P([MASK_l]|unmasked token) 즉 독립 가정, pretraining과 inference(finetunning)시 discrepancy 문제

XLNet은 AR Model과 AE Model의 장점을 합친 버전임.
bidirectional 가능, discrepancy 해결, 독립성 해결

## 2. Permutation Language Model

ex) x1, x2, x3, x4에서 permutation의 경우의수 4!(-> 이것을 다 활용하는 것이 아니라, random sampling 함)  
이 때 x3를 위주로 살펴본다면


| x3 | # | # | # |  
| ---          | ---          | ---          | ---          
| # | x3 | # | # |  
| # | # | x3 | # |  
| # | # | # | x3 |  

즉 permutation을 하게 되면, x3에 대한 양방향의 정보가 다 들어오게 된다. 그래서 bidirectional 문제 해결 완료
(직관적으로 model parameters가 모든 factorization order서 공유된다면, 모든 위치에서의 모든 방향의 정보를 학습할 수 있게 된다 - paper 曰)
그러나 permutation도 문제점이 존재하는데,
![](https://github.com/Chuck2Win/Paper_Review/blob/master/image/CodeCogsEqn%20(1).gif)

we sample factorization order z.
이렇게 AR방식으로 하게 되면, 독립성 가정도 필할 수 있고, Pretrain - Finetunning 불 일치 문제도 해결 可

### Remark on Permutation
original sequence order는 유지(positional encoding 활용)+attention mask를 활용해서 factorization order를 표현 

t-1 까지의 token들로 t 번째 token을 예측함.
예를 들어 설명하면, x4,x2,x3,x1의 것을 sampling했다고 하면, ![$\p(x_z3|x_z1,x_z2)$](https://github.com/Chuck2Win/Paper_Review/blob/master/image/CodeCogsEqn.gif)

기존 transformer가 못하는 사항
1. t번째 token을 예측하는데에 있어서, model은 단순히 t번째 token의 위치만을 고려해야한다.(content은 알면 안된다)
2. 그리고 t번째 token을 예측할 때에 model은 t번째 토큰 이전의 모든 content를 encode 해야한다.
다시 예를 들어서 설명하면, x1,x2,x3,x4라 할 때에, x3을 예측하기 위해서는 x3의 내용을 모르고 위치만을 고려해야 하며, x3을 예측할 때에는 x1,x2의 content를 encode해야한다는 소리.

## 3. Two Stream Self Attention for Target-Aware Representations
![Two Stream Self Attention](https://github.com/Chuck2Win/Paper_Review/blob/master/image/3.png)

![최종목적식](https://github.com/Chuck2Win/Paper_Review/blob/master/image/4.png)

이에 대해서 XLNet은 x_t를 embedding 하기 위해서,
Content Representation for x_z(t) (h_theta(x_z<t)) : content information from x_z(1)~x_z(t) | Position from x_z(1) to x_z(t)
<- Content stream attention (기존의 Transformer의 decoder와 동일)
![Content](https://github.com/Chuck2Win/Paper_Review/blob/master/image/6.png)

Query Representation for x_z(t) (g_theta(x_z<t,zt)) : content information from x_z(1)~x_z(t-1) | Position x_z(t) only (x_z(t)의 content는 접근할 수 없음)
<- Query stream attention 
![Query](https://github.com/Chuck2Win/Paper_Review/blob/master/image/7.png)

example 
input sequence의 order 
| x1 | x2 | x3 | x4 |
| ---          | ---          | ---          | ---          

permutation order 
| x3 | x2 | x4 | x1 |
| ---          | ---          | ---          | ---          


- Content stream attention에서의 mask

| | x1 | x2 | x3 | x4 |
| ---          | ---          | ---          | ---          | ---          
| x1 | | | | |
| x2 | x |  | | x |
| x3 | x | x | | x |
| x4 | x | | | |


- Query stream attention에서의 mask : 자기 자신의 위치와 이전까지의 token의 내용

| | x1 | x2 | x3 | x4 |
| ---          | ---          | ---          | ---          | ---          
| x1 | x | | | |
| x2 | x | x  | | x |
| x3 | x | x | x | x |
| x4 | x | | | x |

그리고 Finetuning할 때엔,query stream을 drop하고, content stream만을 활용한다.

### Partial Prediction
장황하게 표현했지만, optimization difficulty(memory와 speed를 위해서), factorization order에서 마지막 tokens만을 예측함
![partial prediction](https://github.com/Chuck2Win/Paper_Review/blob/master/image/5.png)

예를 들어서 표현하면, factorization order가 x3->x2->x1->x4이면 x1,x4만을 예측함.
여기서 hyper parameter K는 1/K tokens이 예측을 위해서 선택된 것이라고 이해하면 된다.

~근데 여기에서 hyperparameter K는 무엇인지 잘 모르겠음~

### Incorporating Ideas from Transformer-XL
1) relative positional encoding 
[Self-Attention with Relative Position Representation-2018에 처음 나온 idea]
![relative positional encoding](https://github.com/Chuck2Win/Paper_Review/blob/master/image/8.png)

![relative positional encoding](https://github.com/Chuck2Win/Paper_Review/blob/master/image/9.png)

두번째 내가 수기로 정리한 것을 보면, 사실상 이 논문의 핵심과 relative positional encoding은 정리 완료(물론 T5에선 어떤 식으로 하나 알아봐야 할 것 같다)

2) segment recurrence mechanism - 명확히 이해는 되지 않았다. 논의가 필요할 듯.
예를 들어서 s란 sequence에서 총 길이는 1~2T. 이 때 두 개의 segment가 있다고 하자. x_hat : s_1:T, x : s_T+1:2T 
여기에서 z_hat, z respectively는 1~T, T+1~2T의 pemutations이다.
![relative positional encoding](https://github.com/Chuck2Win/Paper_Review/blob/master/image/10.png)

### Relative Segment Encodings
BERT에서는 segment encoding을 활용했는데, XLNet은 Relative segment encoding을 활용한다.
i,j position이 주어졌다면, i와 j가 같은 segment이면 sij=s+, 다르면 sij=s- 
attention weights(내가 수업시간에 배운 용어로는 attention distribution : 즉 softmax를 취한 경우) 
![relative segment_encoding](https://github.com/Chuck2Win/Paper_Review/blob/master/image/11.png)

이 때 q_i는 query vector이고, b는 learnable head specific bias vector (head 마다 다른 bias를 부여하네)
이제 이것을 기존의 normal attention weight에 더한다.
relative segment encoding의 장점은, 일반화 능력이 좋고, finetuning시 2개 이상의 segment가 있을 수 있 때에도 적용이 가능

```
# BERT ways
nn.Embedding(3,128,padding_idx=2) # segment가 A이면 1일테고, 그에 대응되는 벡터값 128개가 있을것이고
```



? 그러면 positional encoding할 때에 자기 자신의 위치만 살려서 더해주고, 나머지 embedding(token embedding)만 mask를 씌우나?


