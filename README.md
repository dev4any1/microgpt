This is a port of the awesome karpathy's example of the generative pretrained transformer, aka GPT

Original python code: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

More info: https://karpathy.github.io/2026/02/12/microgpt/

Inspired by 200 lines of python code, trying to do KISS Java around that go 300 lines keeping conventions and thinking of performance in mind

Some results:


```
num docs: 32033
vocab size: 27
num params: 4192

step    1/1000 | loss 3.2216
step    2/1000 | loss 3.1296
step    3/1000 | loss 3.4762
step    4/1000 | loss 3.2716
step    5/1000 | loss 3.3531
step    6/1000 | loss 3.1744
step    7/1000 | loss 2.8955
step    8/1000 | loss 3.6279
step    9/1000 | loss 3.4292
step   10/1000 | loss 2.7209
...
step  990/1000 | loss 2.6086
step  991/1000 | loss 2.0161
step  992/1000 | loss 2.9062
step  993/1000 | loss 2.2468
step  994/1000 | loss 2.5130
step  995/1000 | loss 2.8132
step  996/1000 | loss 2.9209
step  997/1000 | loss 2.2610
step  998/1000 | loss 1.9681
step  999/1000 | loss 2.6584
step 1000/1000 | loss 2.4970
--- inference (new, hallucinated names) ---
sample  1: shade
sample  2: saeyle
sample  3: jara
sample  4: aryan
sample  5: dlana
sample  6: amela
sample  7: kazd
sample  8: jana
sample  9: keria
sample 10: jahan
sample 11: amara
sample 12: arin
sample 13: paeli
sample 14: bilyle
sample 15: 
sample 16: arian
sample 17: kisha
sample 18: kahan
sample 19: anane
sample 20: hahaiy

```

Awesome visual demo of that algo: https://microgpt.enescang.dev/
