# Bert Emotion Evaluation

# 파이썬 환경 설정
```bash
poetry sync
```

# E2E vs Stepwise
```bash
poetry run e2e
poetry run stepwise
```
결과는 아래와 같이 동일하게 출력됩니다.
```
Test passed for input: I love you so much!.
  - label='love' score=0.9600425362586975
Test passed for input: This is absolutely disgusting!.
  - label='disgust' score=0.8937399983406067
Test passed for input: I'm so happy with my new phone!.
  - label='happiness' score=0.9810405373573303
Test failed for input: Why does this always break?. Expected: anger.
  - label='confusion' score=0.5995238423347473
Test passed for input: I feel so alone right now..
  - label='sadness' score=0.9365758299827576
Test failed for input: What just happened?!. Expected: surprise.
  - label='confusion' score=0.4790430963039398
Test passed for input: I'm terrified of this update failing..
  - label='fear' score=0.897840678691864
Test failed for input: Meh, it's just okay.. Expected: neutral.
  - label='happiness' score=0.5481300354003906
Test failed for input: I shouldn't have said that.. Expected: shame.
  - label='disgust' score=0.5571715235710144
Test failed for input: I feel bad for forgetting.. Expected: guilt.
  - label='sadness' score=0.5780622363090515
Test passed for input: Wait, what does this mean?.
  - label='confusion' score=0.5055228471755981
Test passed for input: I really want that new gadget!.
  - label='desire' score=0.5905269384384155
Test failed for input: Oh sure, like that's gonna work.. Expected: sarcasm.
  - label='happiness' score=0.6655710935592651
```
