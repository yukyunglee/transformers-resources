# 🤗huggingface transformers resources🤗

huggingface transformers를 사용하면서 정리한 자료들을 아카이빙 합니다.

```
# Requirements
transformers==4.5.0
```



## 00 Introduction



### 1) What is Transformers?

<center><img src='https://raw.githubusercontent.com/yukyunglee/yukyunglee.github.io/master/HF.png' width=500>

* **Huggingface Transformer**는 모델을 쉽고 빠르게 생성하고, 학습하고, 배포하기 위해 만들어진 Highlevel Library입니다
* NLP 모델을 쉽고 빠르게 학습할 수 있으며, 다양한 pipeline 및 정형화된 코드를 제공합니다
* Transformers는 Transformer기반의 모델을 쉽고 빠르게 사용할 수 있도록 다양한 기능을 제공합니다



### 2) NLP Modeling with transformers

* NLP 모델은 아래와 같은 Pipeline으로 구성됩니다

  `Input` -> `Tokenization` -> `Model training/Inference` -> `Post-Processing` (task dependent) -> `Output`

* 이러한 pipeline을 모두 scratch부터 구현해보는것은 매우 의미있는 작업입니다
* 하지만 task에 알맞는 모델을 빠르게 구현하기 위해서는 반복적인 작업을 **정형화** 할 필요성이 있습니다
* transformers를 사용함으로서 반복되던 작업을 간편하게 수행할 수 있고, **모델 개발에 초점을 맞추어 개발 및 연구**를 진행할 수 있게됩니다.



### 3) etc.

* 아래의 링크에서 더 자세하고 꼼꼼한 공식 튜토리얼을 공부할 수 있습니다. 

   https://huggingface.co/transformers/notebooks.html

* 아래의 링크에서 huggingface에서 제공하는 라이브러리에 대한 discussion을 살펴볼 수 있습니다. 

  https://discuss.huggingface.co/



## 📂 tutorial

huggingface는 자체적으로 상세한 document를 제공하며, 관련 notebook과 course를 제공합니다. 하지만 보다 쉽게 transformers 를 빠르게 훑어볼 수 있는 내용으로 구성하였으며, 모델링 과정에서 알게된 주의사항을 함께 기술하였습니다. 

#### 01 Basic tutorial

1) Tokenizer 불러오기
2) Config 불러오기
3) Pretrained model 불러오기
4) Huggingface Trainer 사용하기



#### 02 Advanced tutorial

1. Token 추가하기 
   * special token 추가하기
   * 일반 token 추가하기
2. [CLS] output 추출하기
   * 참고 : [CLS] 토큰은 정말 문장을 대표할까 ?



## 📂 note

transformers를 활용하여 모델링을 진행하다 알게된 작고 소중한 메모들을 정리하였습니다.

#### 01 Checkpoint loading

* checkpoint loading시 주의할점

  * model state_dict일부만 loading하기

  

## 📂 modeling

transformers를 활용하여 간단한 nlp 모델을 구현합니다.

##### 01 Named Entity Recognition

* Trainer 사용, config 수정, 함수 분리
* datasets의 [CoNLL-2003 ](https://www.aclweb.org/anthology/W03-0419.pdf) 사용하여 모델링
* eval_f1: 0.93, eval_acc: 0.98

