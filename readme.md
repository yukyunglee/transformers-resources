# ๐คhuggingface transformers resources๐ค

huggingface transformers๋ฅผ ์ฌ์ฉํ๋ฉด์ ์ ๋ฆฌํ ์๋ฃ๋ค์ ์์นด์ด๋น ํฉ๋๋ค.

```
# Requirements
transformers==4.5.0
```



## 00 Introduction



### 1) What is Transformers?

<center><img src='https://raw.githubusercontent.com/yukyunglee/yukyunglee.github.io/master/HF.png' width=500>

* **Huggingface Transformer**๋ ๋ชจ๋ธ์ ์ฝ๊ณ  ๋น ๋ฅด๊ฒ ์์ฑํ๊ณ , ํ์ตํ๊ณ , ๋ฐฐํฌํ๊ธฐ ์ํด ๋ง๋ค์ด์ง Highlevel Library์๋๋ค
* NLP ๋ชจ๋ธ์ ์ฝ๊ณ  ๋น ๋ฅด๊ฒ ํ์ตํ  ์ ์์ผ๋ฉฐ, ๋ค์ํ pipeline ๋ฐ ์ ํํ๋ ์ฝ๋๋ฅผ ์ ๊ณตํฉ๋๋ค
* Transformers๋ Transformer๊ธฐ๋ฐ์ ๋ชจ๋ธ์ ์ฝ๊ณ  ๋น ๋ฅด๊ฒ ์ฌ์ฉํ  ์ ์๋๋ก ๋ค์ํ ๊ธฐ๋ฅ์ ์ ๊ณตํฉ๋๋ค



### 2) NLP Modeling with transformers

* NLP ๋ชจ๋ธ์ ์๋์ ๊ฐ์ Pipeline์ผ๋ก ๊ตฌ์ฑ๋ฉ๋๋ค

  `Input` -> `Tokenization` -> `Model training/Inference` -> `Post-Processing` (task dependent) -> `Output`

* ์ด๋ฌํ pipeline์ ๋ชจ๋ scratch๋ถํฐ ๊ตฌํํด๋ณด๋๊ฒ์ ๋งค์ฐ ์๋ฏธ์๋ ์์์๋๋ค
* ํ์ง๋ง task์ ์๋ง๋ ๋ชจ๋ธ์ ๋น ๋ฅด๊ฒ ๊ตฌํํ๊ธฐ ์ํด์๋ ๋ฐ๋ณต์ ์ธ ์์์ **์ ํํ** ํ  ํ์์ฑ์ด ์์ต๋๋ค
* transformers๋ฅผ ์ฌ์ฉํจ์ผ๋ก์ ๋ฐ๋ณต๋๋ ์์์ ๊ฐํธํ๊ฒ ์ํํ  ์ ์๊ณ , **๋ชจ๋ธ ๊ฐ๋ฐ์ ์ด์ ์ ๋ง์ถ์ด ๊ฐ๋ฐ ๋ฐ ์ฐ๊ตฌ**๋ฅผ ์งํํ  ์ ์๊ฒ๋ฉ๋๋ค.



### 3) etc.

* ์๋์ ๋งํฌ์์ ๋ ์์ธํ๊ณ  ๊ผผ๊ผผํ ๊ณต์ ํํ ๋ฆฌ์ผ์ ๊ณต๋ถํ  ์ ์์ต๋๋ค. 

   https://huggingface.co/transformers/notebooks.html

* ์๋์ ๋งํฌ์์ huggingface์์ ์ ๊ณตํ๋ ๋ผ์ด๋ธ๋ฌ๋ฆฌ์ ๋ํ discussion์ ์ดํด๋ณผ ์ ์์ต๋๋ค. 

  https://discuss.huggingface.co/



## ๐ tutorial

huggingface๋ ์์ฒด์ ์ผ๋ก ์์ธํ document๋ฅผ ์ ๊ณตํ๋ฉฐ, ๊ด๋ จ notebook๊ณผ course๋ฅผ ์ ๊ณตํฉ๋๋ค. ํ์ง๋ง ๋ณด๋ค ์ฝ๊ฒ transformers ๋ฅผ ๋น ๋ฅด๊ฒ ํ์ด๋ณผ ์ ์๋ ๋ด์ฉ์ผ๋ก ๊ตฌ์ฑํ์์ผ๋ฉฐ, ๋ชจ๋ธ๋ง ๊ณผ์ ์์ ์๊ฒ๋ ์ฃผ์์ฌํญ์ ํจ๊ป ๊ธฐ์ ํ์์ต๋๋ค. 

#### 01 Basic tutorial

1) Tokenizer ๋ถ๋ฌ์ค๊ธฐ
2) Config ๋ถ๋ฌ์ค๊ธฐ
3) Pretrained model ๋ถ๋ฌ์ค๊ธฐ
4) Huggingface Trainer ์ฌ์ฉํ๊ธฐ



#### 02 Advanced tutorial

1. Token ์ถ๊ฐํ๊ธฐ 
   * special token ์ถ๊ฐํ๊ธฐ
   * ์ผ๋ฐ token ์ถ๊ฐํ๊ธฐ
2. [CLS] output ์ถ์ถํ๊ธฐ
   * ์ฐธ๊ณ  : [CLS] ํ ํฐ์ ์ ๋ง ๋ฌธ์ฅ์ ๋ํํ ๊น ?



## ๐ note

transformers๋ฅผ ํ์ฉํ์ฌ ๋ชจ๋ธ๋ง์ ์งํํ๋ค ์๊ฒ๋ ์๊ณ  ์์คํ ๋ฉ๋ชจ๋ค์ ์ ๋ฆฌํ์์ต๋๋ค.

#### 01 Checkpoint loading

* checkpoint loading์ ์ฃผ์ํ ์ 

  * model state_dict์ผ๋ถ๋ง loadingํ๊ธฐ

  

#### 02 Tokenizer

* Tokenizer ์ฌ์ฉ์ ๊ณ ๋ คํ๋ฉด ์ข์ ๋ช๊ฐ์ง ํฌ์ธํธ
  * tokenizer.encode์ tokenizer.convert_tokens_to_ids



## ๐ modeling

transformers๋ฅผ ํ์ฉํ์ฌ ๊ฐ๋จํ nlp ๋ชจ๋ธ์ ๊ตฌํํฉ๋๋ค.

##### 01 Named Entity Recognition

* Trainer ์ฌ์ฉ, config ์์ , ํจ์ ๋ถ๋ฆฌ
* datasets์ [CoNLL-2003 ](https://www.aclweb.org/anthology/W03-0419.pdf) ์ฌ์ฉํ์ฌ ๋ชจ๋ธ๋ง
* eval_f1: 0.93, eval_acc: 0.98

