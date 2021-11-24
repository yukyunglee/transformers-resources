## Checkpoint loading



학습된 모델을 불러올 때 학습 환경이 다른 경우 statedict loading 과정에서 에러가 발생할 수 있다. 특히 `AutoModel.from_pretrained()` 에 익숙한 경우 torch로 학습한 모델을 가져올 때 예상하지 못한 에러메세지를 보고 당황하기 쉽기에 관련내용을 정리해보고자 한다.



> 특히 관련 내용은 부스트캠프 멘토링을 진행하며 최소 5번 이상 받은 질문이며, 나 또한 모델링을 진행하며 동일한 에러를 만난적 있다.



#### 01 Model load

기본적으로  `AutoModel.from_pretrained()` 은 hugging face model hub에서 제공되는 모델 이름을 넣어 pretrained model을 Loading 하는것이 일반적이다. 혹은 내가 생성한 모델이 저장된 path를 넣어 저장된 checkpoint를 불러올 수도 있다. 

간혹 finetuned model의 이름을 hub에서 제공하는 이름과 동일하게 세팅하여 저장하는 경우가 있는데, custom name을 주는것이 좋다. 혹은 `local_files_only` option을 `True`로 설정하여 에러를 방지할 수도 있다. 

```python
from transformers import AutoModel
AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
```



#### 02 Loading시 체크해보면 좋을 사항

간혹 학습환경이 상이한 모델을 불러와 loading을 진행하는 경우가 있는데, 이때 몇가지 에러를 만날 수 있다

##### Case 1 : load model의 state dict이 config와 다른 경우

* 가장 흔하게 발생할 수 있는 경우로, 학습 모델이 ddp로 학습되었거나 버전이 다를 경우 model state dict name이 다를 수 있다.

* 이 경우에는 아래와 같은 메세지가 뜨며 key가 match되지않아 모델을 로드 할 수 없다고 나온다

  ~~~
  _IncompatibleKeys(missing_keys=[..],)
  ~~~

* 예를들어 bert.encoder.embeddings ~ vs encoder.embeddings ~ 은 구조가 동일하지만 Key 값이 매치되지 않는 상황이라고 이해할 수 있다
* 구조가 동일한 경우 key 값을 맞춰주는 과정이 필요하며, 대부분 아래와 같은 방법으로 수정하면 된다
* 해당예시는 BERT 모델을 기준으로 한다

~~~python
from transformers import AutoModel,AutoConfig
import glob
import torch
from collections import OrderedDict

model = AutoModel.from_pretrained(Model_name)
state_dict = torch.load(Model_path, map_location='cpu')

keys = list(state_dict.keys())
for key in keys:
    if 'LayerNorm' in key:
        if 'gamma' in key:
            state_dict[key.replace('gamma', 'weight')] = state_dict.pop(key)
        else:
            state_dict[key.replace('beta', 'bias')] = state_dict.pop(key)

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[5:] if k[:5] == 'bert.' else k # remove `bert.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
~~~

* 에러 메세지를 확인해보면 어떤 Key 값에서 문제가 생긴지 명시해주기 때문에 하나씩 키값을 수정하는게 가장 간단한 방법이다
* `strict=False`  을 사용하면 매치되는 키값만 weight를 교체해주기 때문에 비교적 유연하게 값을 넣어줄 수 있으며, 되도록 사용하여 어떤 문제가 있는지 체크하는것이 좋다.
  * 이걸 쓰지 않으면 하나씩 체크하면서 weight가 잘 들어갔는지 확인해줘야한다
  * 혹은 에러가 발생하지 않는 대신 초기화된 가중치가 들어가 의도하지 않은 방향으로 inference를 진행하는 대참사가 일어날 수 있다
    * 대참사가 일어났던 경험이 있다 🤓 (ㅋㅋ)



##### Case 2 load한 모델이 사용하고자 하는 모델과 구조가 다른 경우

>  Some weights of the model checkpoint at 'MODEL_NAME' were not used when initializing BertModel

최근 TAPT, DAPT를 수행한 다음에 모델을 다시 학습하는 경우가 많은데, 자주 만나볼 수 있는 메세지이다. 이건 크게 문제가 되지 않는 메세지이지만, 괜히 터미널에 다음과 같은 메세지가 뜨면 신경 쓰일 수 밖에 없다.

메세지를 잘 읽어보면 큰 문제가 아님을 알 수 있다. BertForPretraining으로 학습하고 나서 downstream task를 수행하면 key match가 되지 않는다는 이야기이고, 어떤 상황에서는 가능하고 어떤상황에서는 문제가 되는지 명확히 말해주고 있다.

> * This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model). 
>
> - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

(transformers는 에러 메세지가 친절하다)