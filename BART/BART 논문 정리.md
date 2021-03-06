# BART 논문 정리

코드 참조

https://github.com/pytorch/fairseq/blob/master/fairseq/models/bart/model.py 

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

  poisson distribution
  $$
  f(x)=\frac{\lambda^{x}e^{-\lambda}}{x!}
  $$
  

  ```python
  if args.mask_length == 'span-poisson':
      _lambda = args.poisson_lambda
  
      lambda_to_the_k = 1
      e_to_the_minus_lambda = math.exp(-_lambda)
      k_factorial = 1
      ps = []
      for k in range(0, 128):
          ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
          lambda_to_the_k *= _lambda
          k_factorial *= (k + 1)
          if ps[-1] < 0.0000001:
              break
      ps = torch.FloatTensor(ps)
      self.mask_span_distribution = torch.distributions.Categorical(ps) # 길이가 0부터 127까지 존재함.
      
  def add_whole_word_mask(self, source, p):
      # 여기서 말하는 whole word mask는 내 짐작에는 Byte Pair Encoding하니깐, 안녕하세요 -> _안녕 _하세요 이렇게 되면
      # 안녕하세요 전체를 mask씌우기 위해서 하는 행위라고 사료됨.
      is_word_start = self.word_starts(source)
      num_to_mask = int(math.ceil(is_word_start.float().sum() * p)) # 올림
      num_inserts = 0
      if num_to_mask == 0:
          return source
  
      if self.mask_span_distribution is not None:  # poisson이라면.
          lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
  		# lengths = [3,0,2,...,1] 이라고 가정하자.	
          # Make sure we have enough to mask
          cum_length = torch.cumsum(lengths, 0) # 누적합
          while cum_length[-1] < num_to_mask:
              lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
              cum_length = torch.cumsum(lengths, 0)
  		# 가령 num_to_mask = 120이라고 한다면
          # cum_length = [3,3,5,...,119]로 종결
          
          # Trim to masking budget ??
          i = 0
          while cum_length[i] < num_to_mask:
              i += 1
          # i는 cum_length의 기록표
          lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1]) # 예시로는 1이 됨.
          num_to_mask = i + 1
          lengths = lengths[:num_to_mask] # [3,0,2,...,1] <- trim됨.
  
          # Handle 0-length mask (inserts) separately
          lengths = lengths[lengths > 0]
          num_inserts = num_to_mask - lengths.size(0)
          num_to_mask -= num_inserts
          if num_to_mask == 0:
              return self.add_insertion_noise(source, num_inserts / source.size(0))
  
          assert (lengths > 0).all()
      else:
          lengths = torch.ones((num_to_mask,)).long()
      assert is_word_start[-1] == 0
      word_starts = is_word_start.nonzero()
      indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
      mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio
  
      source_length = source.size(0)
      assert source_length - 1 not in indices
      to_keep = torch.ones(source_length, dtype=torch.bool)
      is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
      if self.replace_length == 0:
          to_keep[indices] = 0
      else:
          # keep index, but replace it with [MASK]
          source[indices] = self.mask_idx
          source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
  	# check point
      if self.mask_span_distribution is not None:
          assert len(lengths.size()) == 1
          assert lengths.size() == indices.size()
          lengths -= 1
          while indices.size(0) > 0:
              assert lengths.size() == indices.size()
              lengths -= is_word_start[indices + 1].long()
              uncompleted = lengths >= 0
              indices = indices[uncompleted] + 1
              mask_random = mask_random[uncompleted]
              lengths = lengths[uncompleted]
              if self.replace_length != -1:
                  # delete token
                  to_keep[indices] = 0
              else:
                  # keep index, but replace it with [MASK]
                  source[indices] = self.mask_idx
                  source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
      else:
          # A bit faster when all lengths are 1
          while indices.size(0) > 0:
              uncompleted = is_word_start[indices + 1] == 0
              indices = indices[uncompleted] + 1
              mask_random = mask_random[uncompleted]
              if self.replace_length != -1:
                  # delete token
                  to_keep[indices] = 0
              else:
                  # keep index, but replace it with [MASK]
                  source[indices] = self.mask_idx
                  source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
  
              assert source_length - 1 not in indices
  
      source = source[to_keep]
  
      if num_inserts > 0:
          source = self.add_insertion_noise(source, num_inserts / source.size(0))
  
      return source
  ```

  ```python
  # permute sentence 
  '''
  디테일은 잘 모르겠지만
  0 1 2 4 3 1 -> permute시킨다는 것 같음.
  '''
  
  
  def permute_sentences(self, source, p=1.0):
      full_stops = (source == self.full_stop_index) # full stop index : . 구두점임. 끝나는 index라고 생각하면 될 듯.
      # Pretend it ends with a full stop so last span is a sentence
      full_stops[-2] = 1
  
      # Tokens that are full stops, where the previous token is not
      sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2 # ~표시-여집합 개념이네.
      result = source.clone()
  
      num_sentences = sentence_ends.size(0)
      num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0) # 올림
      substitutions = torch.randperm(num_sentences)[:num_to_permute]
      ordering = torch.arange(0, num_sentences)
      ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]
  
      # Ignore <bos> at start
      index = 1
      for i in ordering:
          sentence = source[(sentence_ends[i - 1] if i > 0 else 1):sentence_ends[i]]
          result[index:index + sentence.size(0)] = sentence
          index += sentence.size(0)
      return result
  ```

  

# Model

BART maps a <u>corrupted document</u> to the <u>original document</u>.

6 layers for encoder and decoder

Linear 층을 없애서 BERT에 비해서 10% 정도만 parameter 증가.

![2](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/2.png)



Corruption for text. - 추후 Sentence Permutation과 Text infilling이 결과가 제일 좋았음.

![image-20210303101218454](https://github.com/Chuck2Win/Paper_Review/blob/master/BART/1.png)

[ Text Infilling ]

text span을 [MASK]로 대체 . 이 때 Text span은 말 그대로 text의 선형 결합인 것이고. 이 때 span의 길이는 포아송($\lambda$=3)에서 sampling함.

- 모델이 얼마나 span으로 부터 얼마나 많은 토큰이 사라졌느냐를 학습하게 함.(내 생각엔, span의 길이와 그 내용을 유추하게끔 학습할듯)



# Fine tunning

## Machine Translation(신선)

![4](H:\논문\BART\4.png)

Pretrained 된 BART 모델은 영어를 학습했을터, 논문에서 마냥 체코어를 번역하는 Task를 진행한다면, BART의 Encoder부에 Randomly initiallized Encoder를 추가(Randomized Word Embedding - (역할 : 체코어를 영어로 매칭))

학습 절차

- <u>추가된 Encoder</u>와 <u>BART Positional embedding</u>, 그리고 <u>self attention input projection matrix of BART's encoder first layer</u>만 학습
- 후에 모든 파라미터를 조금만 학습.
