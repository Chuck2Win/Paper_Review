# Ensemble

## Bagging - Back Aggregation

:star: 여러 개의 단일 모델 + `bootstrap` :arrow_forward: 데이터 랜덤 추출 후 모델 훈련 :arrow_right: Voting

- Random Forest

## Boosting

- Xgboost
- Adaboost
- LightGBM



[GBM] - `residual fitting` - parameter : `weak learner`+`weight`

![image-20210325134218256](https://github.com/Chuck2Win/Paper_Review/blob/master/image/gradient_boosting2.jpg)

---

# Tree

https://ratsgo.github.io/machine%20learning/2017/03/26/tree/

$Entropy(A)=\sum{R_i(-\sum_c{p_clog_2p_c})}$

where $R_i$는 비율

:point_right: `Recursive Partitioning` + `Prunning`

비용함수 : $C(T)=Err(T)+\alpha L(T)$

where $Err$ - 오분류율, $L$ - terminal node의 수(`prunning`)

- 독립 변수가 범주형일 경우, 이냐 아니냐 등으로 나눠서 `Entropy` 비교

- 독립 변수가 연속형일 경우, 일단 sort로 해서 배열하고 그 후에 하나를 기준으로 이상 이하로 진행

  ![image-20210325134218256](http://i.imgur.com/XgIfBPX.png)

  

:question: Regression Tree의 작동법 익혀보기

- leaf node는 종속 변수의 평균값을 반환

  ![image-20210325134218256](https://github.com/Chuck2Win/Paper_Review/blob/master/image/regression.png)

- 분할 시  분산 감소를 기준으로 작동함.

  https://www.saedsayad.com/decision_tree_reg.htm

  ![image-20210325134218256](https://github.com/Chuck2Win/Paper_Review/blob/master/image/regression_tree.jpg)

http://www.stat.cmu.edu/~cshalizi/350-2006/lecture-10.pdf 

