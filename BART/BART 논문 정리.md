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

Text infilling은 각각의 결합을 [MASK]의 sequence로 교체해줌(abc -> [MASK] [MASK] [MASK]) : 얼마나 많은 토큰이 mask로 바뀌었는지 예측하게끔.

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

![5](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/5.png)

## Machine Translation(신선)

![4](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/4.png)

Pretrained 된 BART 모델은 영어를 학습했을터, 논문에서 마냥 체코어를 번역하는 Task를 진행한다면, BART의 Encoder부에 Randomly initiallized Encoder를 추가(Randomized Word Embedding - (역할 : 체코어를 영어로 매칭))

학습 절차

- <u>추가된 Encoder</u>와 <u>BART Positional embedding</u>, 그리고 <u>self attention input projection matrix of BART's encoder first layer</u>만 학습
- 후에 모든 파라미터를 조금만 학습.  
- (여기에서 말하는 Positional embedding은 )
```
```

# Comparison with other models  
1M step 학습 + data : combinations of books & Wikipedia data.  
Language Model: GPT(cross attention이 없는 BART의 Decoder와 유사.)    
Permuted Language Model: XL-Net(relative positional embedding 과 attention across segments를 활용 x)    
Masked Language Model: BERT(15% Mask 씌우고, 각각의 MASK를 independent하게 예측하게끔 함)    
Multitask Masked Language Model: UniLM  
Masked Seq-to-Seq: MASS  
- Permuted Language LM, Masked LM, Multitask Masked LM -> 2 stream attention을 활용함.(XLNET은 알겠다만, BERT는?)      
(comment : UniLM과 MASS에 대해선 컨셉은 파악해야겠음)  
실험 방식)  
1) 
# Task 
- SQuAD : Extractive QA task  
- MNLI : Bitext Classification task (두 문장의 의미적인 관계 분류)  - ELI5 : Abstractive summary task
- Xsum : Abstractive summary task
- ConvAI2 : Persona를 활용한 대화 생성
- CNN/DM : 뉴스 요약 task.  
# Results  
https://dladustn95.github.io/assets/images/bart_figure7.png
