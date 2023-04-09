# AuToeic: Auto-making exams of TOEIC part 1
토익 LC Part 1 문제를 출제해주는 모델

## DSL 23-1 Modelling Project H조
**팀명** : Hairy Potatoes  
**팀원** : 엄소은, 이재우, 송규원, 이성균

## Overview
---


![05](https://user-images.githubusercontent.com/121658932/230769569-669cdbcc-230c-4b0f-9273-5f721c48ea35.jpg)

**1. 개요**
- Background:
  - CV와 NLP를 한번에 구현하면서도 유의미한 결과물을 생성해내는 알고리즘을 만들어보려고 함
  - 토익 part 1은 사진을 보고 가장 적절하게 묘사한 문장을 고르는 문제
  - 저작권 문제 없이 토익의 독자적인 사진을 찍고 문제를 출제하는 데에 많은 비용이 들 것으로 예상함
- 기획 내용
  - 문제를 출제하기 위해 필요한 사진을 직접 생성
  - 사진에 적절한 정답 선지를 생성
  - 너무 난이도가 낮아지지 않도록 매력적인 오답 선지들도 생성
  - 문제 난이도가 적절한지 객관적으로 평가
 - 해결 방법
  - 토익 문제의 사진을 생성하기 위한 image-caption paired dataset을 확보
  - 토익 실제 part 1 기출문제를 확보하고 전처리 진행
  - Stable Diffusion을 활용하여 실제로 찍은 것 같은 적절한 사진을 생성
  - BLIP을 활용하여 이전에 생성해낸 사진을 묘사하는 정답 선지와 관련된 오답 선지를 생성
  - CLIP의 zero-shot prediction을 활용하여 사진과 선지 4개와의 유사도(=정답의 확률)를 체크하여 문제가 잘 출제되었는지 평가
 
 **2. 데이터 확보**
 - image와 caption이 쌍으로 맺어진 데이터(common dataset)가 필요
    - Facebook/Winoground (400 rows)
    - Facebook/pmd (70,000,000 rows)
    - Michelecafagna26/hl (149997images & 134973 high-level captions)
    - Action-Effect (140 verb-nounpairs & 10 images and captions)
 - 실제 토익 part 1 기출문제의 사진, 정답 선지, 오답 선지가 필요
    - image-text pair 500개를 직접 수집
    - 그 중 480개는 training, 20개는 testing에 활용
    - 정답 선지를 만들어내는 BLIP 모델에는 정답 선지를 text로, 오답 선지를 만들어내는 BLIP 모델에는 오답 선지를 text로 활용 (아래 stage 2에서 추가 설명)
  
**3. 데이터 전처리**
- common dataset은 그대로 읽음
- 기출문제의 사진, 정답 선지, 오답 선지는 preprocessor를 통해 embedding



**4. Pipeline**

![09](https://user-images.githubusercontent.com/121658932/230769577-e6eb7037-d223-4bf6-bb21-2de7cef9fdfa.jpg)

**5. 전이학습 주요 내용**

**1) Stage 1 : Stable Diffusion 1.5**

![10](https://user-images.githubusercontent.com/121658932/230769579-80a30828-c9ec-4c14-a1dc-f368f73b53c1.jpg)

- 출제를 원하는 주제어를 입력하면 그에 알맞는 고해상도의 사진을 생성해야 함
  - e.g. `Black and white photo, Some bicycles have been left unattained`
  - 원래 컬러 사진이 생성되는데, 토익 시험지는 항상 흑백이므로 흑백 사진을 만들어내기 위해 `Black and white photo`를 입력해줌
- 

**2) Stage 2 : BLIP의 Capfilt**

![11](https://user-images.githubusercontent.com/121658932/230769581-80c94943-9ea5-4352-9364-96a2ac7d8eaa.jpg)

![12](https://user-images.githubusercontent.com/121658932/230769582-bec8bf4c-7611-48c1-af63-0e464c0a2051.jpg)

- 

**3) Stage 3 : CLIP의 zero-shot prediction**

![14](https://user-images.githubusercontent.com/121658932/230769586-d3833a2f-b75b-414b-af22-88538efa9b43.jpg)

- CLIP은 본래 이미지와 텍스트 각각의 embedding vector 간 코사인 유사도를 구하여 학습한 모델로, 새로운 이미지 1개를 입력하고 상응하는 후보 텍스트 n개를 입력하면 그중 가장 연관성이 높은 텍스트를 반환해주는 classification model
- 이때 연관성이 높은지를, embedding vector 간 코사인 유사도를 구한 후 softmax 함수를 씌워서 나온 값(확률)을 비교해서 판단함
- 이러한 원리에 착안하여, stage 1과 2에서 구한 사진의 embedding vector와 정답 선지 1개, 오답 선지 3개의 embedding vector 간의 코사인 유사도를 구한 후 softmax 함수를 씌우면 이미지와 선지 간 적중률(정답률)이 도출될 것으로 예상함 (Dataframe 형태로 반환)
- 실제로 이 정답률이 90% 이상으로 나오는 경우만 유의미한 문제로 판단하고, 이에 미치지 못하는 문제의 경우 제외할 것으로 기대됨

# Model
---
## Stage 1. Prompt-to-image

- Stable Diffusion 1.5 모델 활용
- 글 (Prompt)을 입력하면 Stable Diffusion 1.5을 통해 image가 생성됨
  - ex) 'Black and white photo, Some bicycles have been left unattained'라는 문장 입력하면 자전거가 벽에 기대어 있는 흑백 이미지 생성
![image](https://user-images.githubusercontent.com/108797646/230771341-c4253566-4377-4418-a296-48f35d020ed6.png)

## Stage 2. Image-to-prompt
- BLIP 모델 활용
- 이미지(Image)를 input으로 넣으면 BLIP을 통해 prompt 생성

### 1. 정답 선지 생성 (1개)
- 이미지 & 정답 선지 paired dataset 활용
- 토익 기출문제/모의고사 데이터에서 이미지와 정답 선지 pair를 BLIP 모델에 학습시킴
- Stage 1에서 생성된 이미지를 input으로 넣으면 정답 선지를 생성하는 BLIP 모델(BLIP-c)이 정답 선지 prompt를 생성
  - ex) Stage 1에서 생성된 이미지의 정답 선지로 'Some bicycles are parked near a wall' prompt 생성

### 2. 오답 선지 생성 (3개)
- 오답 선지는 총 3개 생성: 매력적인 오답 1개, 일반 오답 2개
#### 1) 매력적인 오답 1개: 정답 선지와 특정 키워드는 동일/비슷하되, 오답인 선지
   - 이미지 & 오답 선지 paired dataset 활용: 오답 선지 중 가장 매력적인 오답 선별한 dataset
   - 토익 문제 이미지와 오답 중 가장 매력적인 오답 선지 pair를 BLIP 모델에 학습시킴
   - Stage 1에서 생성된 이미지를 input으로 넣으면 매력적인 오답 선지를 생성하는 BLIP 모델(BLIP-w)이 오답 선지 생성
    - ex) Stage 1에서 생성된 이미지의 매력적인 오답 선지로 'A bike is leaning against a column' prompt 생성
        
#### 2) 일반 오답 2개
   - Action-effect 데이터 활용 
    - Caption & Negative_image_list (대상 (사물 or 사람)은 같으나 해당 caption으로 설명이 되지 않는 image) pair data로 구성
   - Action-effect 데이터로 학습시킨 후, Stage 1에서 생성된 이미지를 input으로 넣으면 랜덤한 오답 선지 2개 생성
    - ex) Stage 1에서 생성된 이미지의 오답 선지로 1) 'The door is no longer on the hinges', 2) 'A person sits on a bicycle and makes it move' prompts 생성

![image](https://user-images.githubusercontent.com/108797646/230771911-71e957f3-ac1b-4c26-883e-2b8dd40d5447.png)

## Stage 3. Validation
- CLIP 모델을 활용한 문제 평가
- CLIP 모델이 이미지를 보고 여러 개 보기 중 가장 이미지를 잘 설명하는 text를 골라줌
  - ex) image 2개와 보기 8개를 input으로 넣으면 각 이미지에 가장 적합한 보기를 퍼센트를 부여하여 골라줌
- 비슷한 선지는 50:50 확률을 부여하고, 중복 정답 처리가 될 가능성이 있는 선지들도 10 퍼센트 이상의 확률이 부여됨
- → 90% 이상의 확률이 부여된 선지만 정답 선지로 인정

![image](https://user-images.githubusercontent.com/108797646/230772471-a84ea54d-56e4-4545-a836-b6f06863a43b.png)

# Trial And Error
#### 1) Multiple Datasets
  - 데이터셋의 종류와 조합(비례배분)에 따라 모델의 성능의 차이가 많이 남
  
#### 2) GPT2 모델  
  - 단어 1개 / 단어 뭉치 / 문장 입력하면 이어서 문장 작성 (동일 주제 문장 생성) 가능
  - 하지만 유사한 의미의 문장은 생성해내지 못함
  - → BLIP으로 오답 선지 학습시켜 문장 생성해냄

# 프로젝트의 가치
- CV + NLP를 종합적으로 구현해볼 수 있는 기회
- 우리의 목적에 맞게 fine-tuning 해볼 수 있는 좋은 기회
- Prompt에서 image로, image에서 다시 prompt로 변환하는 과정을 거치는 모델 생성 
- TOEIC LC Part 1 문제를 자동으로 출제하는 모델 생성


# End-to-End Inference
---
위에 설명처럼, 원하는 상황을 기반으로 토익 LC Part1 이미지를 만들고, 선지 4개를 만들어주는 end-to-end pipeline 코드를 구성

### Dependencies
- Python 3.9
- Pytorch 1.12.0
- dependencies in requirements.txt

### How to run
1. Clone the repository
~~~
git clone https://github.com/ddoddii/DSL-23-1-modeling-AuToeic.git
~~~

2. Install pytorch and other dependencies
~~~
pip install -r requirements.txt
~~~

3. Run with options
-For example,
~~~
python main.py -c captions.txt
~~~
`captions`
- Text you want to generate an image from
- Can be more than one sentence
- Generates more detailed images with detailed descriptions
- ex. black and white photeo, two man looking at a computer

## Result
---
1. Image Generate
- Result: imagegen 폴더 내에 caption 기반으로 만들어진 이미지 img_{timestamp}.jpg 생성

2. Toeic Problem Generate
- Result: 만들어진 이미지와 4개의 선지가 포함된 Toeic Problem_{timestamp}.jpg 생성 

3. Clip Evaluation
- 만들어진 문제를 Clip 모델을 이용하여 평가한 dataframe 으로,clipevaluate.png 생성

## File Description
---
### main
- `main.py` : 전체 end-to-end pipeline 수행
- `text2img.py` : text 를 받아서 image 만들어서 저장 
- `img2toeic.py` : 생성한 image 로 토익 image 생성
    - `loadimages` : load images
    - `CreateToeic`: Create Toeic Image, save "Toeic problem_{timestamp}.jpg"
- `clip.py`: Toeic Problem evaluation by clip model, save "clipevaluate.png"

### model
- `model_R`: 정답 선지로 훈련시킨 모델
- `model_W1`, `model_W2`, `model_W3` : 오답 선지로 훈련 시킨 모델

### data
- train: Toeic LC Part1 이미지-선지 데이터셋 (480개)
    - `metadata.jsonl`: image-text paired data, description about images
