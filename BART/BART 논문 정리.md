# BART 논문 정리

코드 참조

https://github.com/pytorch/fairseq/blob/master/fairseq/models/bart/model.py  
https://huggingface.co/transformers/model_doc/bart.html  
개인적으로 huggingface의 그것이 더 가독성이 좋은듯.  

![contextual word embedding](https://github.com/Chuck2Win/Paper_Review/blob/master/image/diagram.png)  


Denosing Autoencoder  

seq2seq 모델이라고 일컬음.(XLNet에서)  

- 창원의 정리

  | Model | Transformer         | Bi directional                | 비고          |
  | ----- | ------------------- | ----------------------------- | ------------- |
  | BERT  | Transformer Encoder | O                             | MASK LM + NSP, AE |
  | GPT   | Transformer Decoder | X                             | LM, AR            |
  | BART  | Transformer         | Encoder부 : O, Decoder 부 : X | MASK LM + LM, Seq2Seq  |

  ![BART 비교](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/BART%20%EB%B9%84%EA%B5%90.png)

Encoder의 input : Corrupted Text(Text infilling & Sentence Permutation)

Text infilling은 각각의 결합을 [MASK]로 교체해줌(abc -> [MASK]) : 얼마나 많은 토큰이 mask로 바뀌었는지 예측하게함.  

Decoder의 input :  Original Text



![3](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/3.png)

- 그림에서 .이 문장이 끝나는 지점.

  

https://github.com/pytorch/fairseq/blob/aa79bb9c37b27e3f84e7a4e182175d3b50a79041/fairseq/data/denoising_dataset.py

- denosing data set
 

# Model

BART maps a <u>corrupted document</u> to the <u>original document</u>.

6 layers for encoder and decoder

Linear 층을 없애서 BERT에 비해서 10% 정도만 parameter 증가.

![2](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/2.png)



Corruption for text. - 추후 Sentence Permutation과 Text infilling이 결과가 제일 좋았음.



text span을 [MASK]로 대체 . 이 때 Text span은 말 그대로 text의 선형 결합인 것이고. 이 때 span의 길이는 포아송(lambda=3)에서 sampling함.

- 모델이 얼마나 span으로 부터 얼마나 많은 토큰이 사라졌느냐를 학습하게 함.(내 생각엔, span의 길이와 그 내용을 유추하게끔 학습할듯)



# Fine tunning
## Machine Translation(신선)

![4](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/4.png)

Pretrained 된 BART 모델은 영어를 학습했을터, 논문에서 마냥 체코어를 번역하는 Task를 진행한다면, BART의 Encoder부에 Randomly initiallized Encoder를 추가(Randomized Word Embedding - (역할 : 체코어를 영어로 매칭))

학습 절차

- <u>추가된 Encoder</u>와 <u>BART Positional embedding</u>, 그리고 <u>self attention input projection matrix of BART's encoder first layer</u>만 학습
- 후에 모든 파라미터를 조금만 학습.  
- (여기에서 말하는 Positional embedding은 )
```{.python}
# Attention is all you need에서의 positional encoding
'''
PE(pos,2*i)=sin(pos/10000**(2*i/d_model))
PE(pos,2*i+1)=cos(pos/10000**((2*i+1)/d_model))
이때 pos는 seq len에서의 위치이고, d_model은 model의 크기, 2i, 2i+1에 들어가는 것은 각 dim임(0~d_model)
'''
position = torch.arange(0, max_len).unsqueeze(1) # max_len,1
div_term = 10000**(torch.arange(0,d_model)/d_model).unsqueeze(0) # 1, d_model
pe = position / div_term
pe[:, 0::2] = torch.sin(pe[:, 0::2])
pe[:, 1::2] = torch.cos(pe[:, 1::2])

# BART positional embedding은 말그대로 max len과 d_model 간의 nn.Embedding
positional_embedding  = nn.Embedding((max_len,d_model)) # 이런식
```

# Comparison with other models  
1M step 학습 + data : combinations of books & Wikipedia data.  
Language Model: GPT(cross attention이 없는 BART의 Decoder와 유사.)    
Permuted Language Model: XL-Net(relative positional embedding 과 attention across segments를 활용 x)    
Masked Language Model: BERT(15% Mask 씌우고, 각각의 MASK를 independent하게 예측하게끔 함)    
Multitask Masked Language Model: UniLM  
Masked Seq-to-Seq: MASS  
- Permuted Language LM, Masked LM, Multitask Masked LM -> 2 stream attention을 활용함.(XLNET은 알겠다만, BERT는?)      
_(comment : UniLM과 MASS에 대해선 컨셉은 파악해야겠음)_
실험 방식)  

# Task 
- SQuAD : Extractive QA task  
- MNLI : Bitext Classification task (두 문장의 의미적인 관계 분류)  - ELI5 : Abstractive summary task
- Xsum : Abstractive summary task
- ConvAI2 : Persona를 활용한 대화 생성
- CNN/DM : 뉴스 요약 task.  
크게 정리하면, Descriminative & Generation Task로 구성.  

# Results  
![4](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/table1.png)  
1) Performance of pre-training methods varies significantly across tasks.
2) Token masking is crucial
3) Left to right pretraining improves generation
4) Bidirectional encoders are crucial for SQuAD
5) The pretraining objective is not the only important factor  
  - 본 논문에선 XLNet과는 조금 다르게, relative position embedding 또는 segment level recurrunce를 빼줌(즉, 모델의 architecture도 중요하다..)  
6) Pure language models perform best on ELI5.
  - ELI5가 outlier라고 칭하고 있음.  
In conclusion : BART achieves the most consistently strong performance.  

# Large scale Pretraining Experiments  
12 layers for encoder and decoder, hidden size 1024, Batch size : 8000(부럽다), train 500,000 steps  
GPT2와 유사하게 BPE Encoding을 활용함. Text infilling과 sentence permutation을 함. 30%의 tokens를 MASK.  
마지막 10%의 training step에서는 drop out을 안하게함.  
dataset : 160GB의 news, books, stories and web text.  
![4](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/table2.png)  

# Conclusion  
Discriminative 에서 RoBerTa와 유사한 성능을 냈고, generation task에서는 sota 성능을 달성하였음  
