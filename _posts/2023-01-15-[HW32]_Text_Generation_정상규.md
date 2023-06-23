---
layout: single
title:  "jupyter notebook 올리기"
categories: python
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


 ##**10. Text Generation**

1. Pretrained Model을 이용해 Text를 generation 하는 model을 구현합니다.

2. 실제 데이터셋을 가지고 모델을 학습해봅니다.

3. 다양한 decoding strategy를 이용하여 text를 생성해봅니다.



```python
!pip install transformers==4.9.2
```

<pre>
Requirement already satisfied: transformers==4.9.2 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (4.9.2)
Requirement already satisfied: pyyaml>=5.1 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (5.3.1)
Requirement already satisfied: regex!=2019.12.17 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (2020.10.15)
Requirement already satisfied: sacremoses in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (0.0.43)
Requirement already satisfied: tqdm>=4.27 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (4.49.0)
Requirement already satisfied: filelock in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (3.0.12)
Requirement already satisfied: numpy>=1.17 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (1.19.2)
Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (0.10.3)
Requirement already satisfied: packaging in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (20.4)
Requirement already satisfied: requests in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (2.24.0)
Requirement already satisfied: huggingface-hub==0.0.12 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from transformers==4.9.2) (0.0.12)
Requirement already satisfied: click in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from sacremoses->transformers==4.9.2) (7.1.2)
Requirement already satisfied: six in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from sacremoses->transformers==4.9.2) (1.15.0)
Requirement already satisfied: joblib in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from sacremoses->transformers==4.9.2) (0.17.0)
Requirement already satisfied: pyparsing>=2.0.2 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from packaging->transformers==4.9.2) (2.4.7)
Requirement already satisfied: chardet<4,>=3.0.2 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from requests->transformers==4.9.2) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from requests->transformers==4.9.2) (1.25.11)
Requirement already satisfied: certifi>=2017.4.17 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from requests->transformers==4.9.2) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from requests->transformers==4.9.2) (2.10)
Requirement already satisfied: typing-extensions in /home/jiminhong/anaconda3/lib/python3.8/site-packages (from huggingface-hub==0.0.12->transformers==4.9.2) (3.7.4.3)
</pre>

```python
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
```


```python
import numpy as np
import random

def set_seed(random_seed):
    torch.random.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
```


```python
set_seed(777)
```


```python
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
```

<pre>
The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'GPT2Tokenizer'. 
The class this function is called from is 'PreTrainedTokenizerFast'.
</pre>

```python
text = '근육이 커지기 위해서는'
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids]),
                           max_length=50,repetition_penalty=1.0,
                           top_k=5,
                           temperature=1.0,                          
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id)
```


```python
generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 규칙적인 생활습관이 중요하다.
특히, 아침식사는 단백질과 비타민, 무기질 등 영양소가 풍부한 음식을 골고루 섭취하는 것이 좋다.
또한 규칙적인 운동은 근육을 강화시켜주는 효과가 있다.
특히, 아침식사는 단백질과 비타민
</pre>
## Likelihood-based Decoding



* Greedy Search

* Beam Search



```python
def greedy(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)
```


```python
class SamplerBase:
    def __init__(self, model, seq_length):
        self.model = model
        self.seq_length = seq_length

    def sample(self, inps, past):
        return NotImplementedError
```


```python
from copy import deepcopy
```

## Greedy Search Decoding



```python

def greedy(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)

class GreedySampler(SamplerBase):
    def __init__(self, model, seq_length, top_whatever, stochastic=False, temperature: float = 1.0):
        """
        :param model:
        :param seq_length:
        :param stochastic: choice [top_k,top_p] if True
        """
        super(GreedySampler, self).__init__(model, seq_length)

        self.sampling = greedy

    @torch.no_grad()
    def sample(self, inps):
        inps=torch.LongTensor([inps])        
        context = inps
        generated = deepcopy(inps)
        past = None

        for t in range(0, self.seq_length):
            out = self.model(context, past_key_values=past)
            lm_logits,past= out["logits"],out["past_key_values"]
            
            lm_logits = lm_logits[:, -1]
            
            context = self.sampling(lm_logits)
            generated = torch.cat([generated, context], dim=-1)

        return generated
```

## Hugging face Library



```python
gen_ids = model.generate(torch.tensor([input_ids]),max_length=34)

generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 규칙적인 생활습관이 중요하다.
특히, 아침식사는 단백질과 비타민, 무기질 등 영양소가 풍부한 음식을 골고루 섭취하는 것이 좋다.
또한 규칙적인
</pre>
## 비교해보기



```python

sampler=GreedySampler(model,30,1)

sampled_ids=sampler.sample(input_ids)

generated = tokenizer.decode(sampled_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 규칙적인 생활습관이 중요하다.
특히, 아침식사는 단백질과 비타민, 무기질 등 영양소가 풍부한 음식을 골고루 섭취하는 것이 좋다.
또한 규칙적인
</pre>
## Beam Search Decoding



```python
class BeamSampler(SamplerBase):
    def __init__(self, model, seq_length, beam_size: int = 3, temperature: float = 1.0):
        """
        no version on stochastic mode
        :param model:
        :param seq_length:
        :param top_whatever: int as beam_size
        """
        super(BeamSampler, self).__init__(model, seq_length)
        self.temperature = temperature
        # if not isinstance(beam_size, int):
        #     raise ValueError
        self.beam_size = beam_size
        self.sampling = greedy

    def _set_start_sequence(self, inps):
        batch, seq_lens = inps.size()
        res = inps[:, None].repeat(1, self.beam_size, 1)  # [batch, beam, l]
        res.view(-1, seq_lens)

        return res.view(-1, seq_lens)

    @torch.no_grad()
    def sample(self, inps):
        inps=torch.LongTensor([inps])
        n_batch, seq_length = inps.size()
        context = self._set_start_sequence(inps)
        generated = deepcopy(context)
        past = None

        probs = torch.zeros([n_batch * self.beam_size]).to(context.device)
        for t in range(0, self.seq_length):
            out = self.model(context, past_key_values=past)
            lm_logits,past= out["logits"],out["past_key_values"]
#             lm_logits, past = self.model(context, past=past)
            
            lm_logits = lm_logits[:, -1]

            context, probs, past, generated = self.beam_sample(lm_logits, probs, t, past, generated)

        return generated.cpu()[:, 0], probs

    def beam_sample(self, logits, probs, time_step, past, generated):

        if time_step == 0:
            logits = logits.view(-1, self.beam_size, logits.size()[-1])
            probs, preds = self.beam_start(logits, probs)
            generated = torch.cat([generated, preds], dim=-1)

        else:
            logits = logits.view(-1, self.beam_size, logits.size()[-1])
            probs, preds, past, generated = self.beam_continue(logits, probs, past, generated)

        return preds.view(-1, 1), probs, past, generated

    def beam_start(self, logits, probs):
        logits = logits / self.temperature
        p, i = torch.topk(torch.log_softmax(logits, -1), self.beam_size, -1)  # [batch, beam_size]
        i = i.view(-1, self.beam_size, self.beam_size)[:, 0, :].contiguous().view(-1, 1)
        p = p.view(-1, self.beam_size, self.beam_size)[:, 0, :].contiguous().view(-1, 1)

        probs = probs + p.view(-1)

        return probs, i

    def beam_continue(self, logits, probs, past, generated):
        bs = logits.size(0)
        generated = generated.view(bs, self.beam_size, -1)

        current_p, indexes = torch.topk(torch.log_softmax(logits, -1), self.beam_size,
                                        -1)  # [batch_size, beam_size, beam_size]
        probs = probs.view(bs, -1).unsqueeze(-1) + current_p
        new_probs = probs.view(bs, -1)

        probs, ni = new_probs.topk(self.beam_size, -1)
        sampled = indexes.view(bs, -1).gather(1, ni)  # [batch, beam]
        group = ni // self.beam_size
        ind = torch.arange(bs)[:, None], group
        generated = generated[ind]
        bs_beam = past[0][0].size(0)

        n_head, seq_len, hidden_size = past[0][0].size()[1:]

        past = [
            (k.view(bs, self.beam_size, n_head, seq_len, hidden_size)[ind].view(bs_beam, n_head, seq_len, hidden_size),
             v.view(bs, self.beam_size, n_head, seq_len, hidden_size)[ind].view(bs_beam, n_head, seq_len, hidden_size)) \
            for k, v in past]

        # sampled = indexes.view(bs, -1).gather(1, ni)
        generated = torch.cat([generated, sampled[:, :, None]], -1)

        return probs, sampled.view(-1)[:, None], past, generated
```

## Huggingface 정답



```python
gen_ids = model.generate(torch.tensor([input_ids]),max_length=34,
                         num_beams=3,temperature=2.0)
generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 면역력을 높여야 한다.
면역력을 높여야 하는 이유다.
면역력을 높여야 하는 이유다.
면역력을 높여야 하는 이유다.
면역력을 높여
</pre>
## 비교해보기



```python
sampler=BeamSampler(model,30,3,temperature=2.0)

sampled_ids=sampler.sample(input_ids)[0]

generated = tokenizer.decode(sampled_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 면역력을 높여야 한다.
면역력을 높여야 하는 이유다.
면역력을 높여야 하는 이유다.
면역력을 높여야 하는 이유다.
면역력을 높여
</pre>
# Stochastic-based Decoding

* Top-k sampling



* Top-p sampling


## Hugging face Library



```python
## top-k sampling

gen_ids = model.generate(torch.tensor([input_ids]),max_length=34,
                         do_sample=True,top_k=5,temperature=1.0)
generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 영양을 공급받는 것이 중요하다.
영양은 바로 우리 몸을 튼튼하게 만들어준다.
비타민의 경우 우리 몸의 혈액 순환과 세포막의 에너지 생성을 도와주기
</pre>
* top-p sampling



```python
## top-k sampling

gen_ids = model.generate(torch.tensor([input_ids]),max_length=34,
                         do_sample=True,top_p=0.1,temperature=1.0)
generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 영양소가 풍부한 음식을 섭취하는 것이 중요하다.
특히, 비타민C가 풍부한 음식은 비타민C가 풍부한 음식을 섭취하는 것이 좋다.
비타민C는
</pre>

```python
def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1, None]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
```


```python
def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch = logits.size(0)
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    a = torch.arange(0, batch).to(logits.device)
    b = torch.max(torch.sum(cumulative_probs <= p, dim=-1) - 1, torch.Tensor([0]).long().to(logits.device))
    min_values = sorted_logits[a, b].to(logits.device)

    return torch.where(
        logits < min_values[:, None],
        torch.ones_like(logits) * -1e10,
        logits,
    )
```


```python
class StochasticSampler(SamplerBase):
    def __init__(self, model, seq_length, top_whatever, stochastic_func, temperature: float = 1.0):
        """
        :param model:
        :param seq_length:
        :stochastic_func
        """
        super(StochasticSampler, self).__init__(model, seq_length)

        self.temperature = temperature
        self.top_whatever=top_whatever
        self.sampling = stochastic_func
        

    @torch.no_grad()
    def sample(self, inps):
        inps=torch.LongTensor([inps])
        context = inps
        generated = deepcopy(inps)
        past = None

        for t in range(0, self.seq_length):
            out = self.model(context, past_key_values=past)
            lm_logits,past= out["logits"],out["past_key_values"]            
            lm_logits = lm_logits / self.temperature
            lm_logits = lm_logits[:, -1]
            masked_lm_logits = self.sampling(lm_logits, self.top_whatever)
            context = torch.multinomial(torch.softmax(masked_lm_logits, -1), 1)
            generated = torch.cat([generated, context], dim=-1)

        return generated
```


```python
sampler=StochasticSampler(model,30,10,top_k_logits)
sampled_ids=sampler.sample(input_ids)
generated = tokenizer.decode(sampled_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 무엇보다 영양섭취가 중요하다.
또 영양섭취에 대한 관심이 높아지면 다이어트나 운동 등 건강관리를 위한 각종 식이요법의 개발이 필요하다.
영양과다
</pre>

```python
sampler=StochasticSampler(model,30,0.5,top_p_logits)
sampled_ids=sampler.sample(input_ids)
generated = tokenizer.decode(sampled_ids[0,:].tolist())
print(generated)
```

<pre>
근육이 커지기 위해서는 원치 않는 곳에 무리하게 자꾸만 손이 가는 것이 가장 큰 이유다.
이럴 때면 엉덩이를 많이 쓸어 올리는 것보다 가볍게 하는
</pre>
## 참고 자료


https://huggingface.co/transformers/v2.6.0/quickstart.html#using-the-past



https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/search.py#L103



https://jeongukjae.github.io/posts/cs224n-lecture-15-natural-language-generation/

