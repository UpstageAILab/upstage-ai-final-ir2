# Scientific Knowledge Question Answering
## IR 2조

| ![김태한](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김소현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김준호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최장원](https://avatars.githubusercontent.com/u/156163982?v=4) |  
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |  
|   [김태한](https://github.com/UpstageAILab)   |    [김소현](https://github.com/UpstageAILab)       |   [김준호](https://github.com/UpstageAILab)     |            [최장원](https://github.com/UpstageAILab)             |   
|    팀장, 리서치, 데이터생성, 모델링   |      리서치, 데이터생성, 모델링          |           리서치, 데이터생성, 모델링         |     리서치, 데이터생성, 모델링    |  

## 0. Overview
### Environment
- AMD Ryzen Threadripper 3960X 24-Core Processor
- NVIDIA GeForce RTX 3090
- CUDA Version 12.2

### Requirements
- openai==1.7.2
- elasticsearch==8.8.0
- sentence_transformers==2.2.2
- wandb

## 1. Competiton Info

### Overview

- 과학 지식 질의 응답 시스템 구축
- LLM은 최근 좋은 성능을 보이지만 Hallucination과 Knolwedge Cut-off 현상이 있다.
- 위 단점을 극복하기 위해서 RAG (Retrieval Augmented Generation)를 도입한다.

- LLM의 등장 이후 여러 산업 분야에서 지식을 다루는 업무들이 점점 고도화되고 있습니다.
  특히 정보를 찾기 위해 검색엔진의 입력창에 키워드를 입력하고 결과를 확인하고 원하는 정보가 없으면 다른 키워드로 다시 검색하기를 반복하는 번거로운 과정을 이제 더이상 자주 할 필요가 없어졌습니다.
  이제 LLM한테 물어보면 질문의 의도까지 파악해서 필요한 내용만 잘 정리해서 알려 줍니다.  
  ![image](https://github.com/UpstageAILab/upstage-ai-final-ir2/blob/main/docs/prompt_0.png)  
  
- 그렇지만 LLM이 가진 근본적인 한계도 있습니다. 먼저, 정보라는 것은 의미나 가치가 시간에 따라 계속 변하기 때문에 모델이 이를 실시간으로 학습하기 힘들고 이 때문에 아래 예시처럼 knowledge cutoff 가 자연스럽게 발생합니다.  
  ![image](https://github.com/UpstageAILab/upstage-ai-final-ir2/blob/main/docs/prompt_1.png)  

- 그리고 LLM이 알려주는 지식이 항상 사실에 기반한 것이 아닌 경우가 종종 있습니다. 특히 특정 도메인이나 문제 영역은 매우 심각한 거짓 정보들을 생성해 내곤 합니다. 아래 예시에서 추천하는 맛집들은 모두 실재하지 않는 장소들입니다.  
  ![image](https://github.com/UpstageAILab/upstage-ai-final-ir2/blob/main/docs/prompt_2.png)  
  
- 이러한 환각 현상은 메타인지를 학습하지 않은 LLM의 근본적인 한계라 볼 수 있습니다.
  모델은 학습 과정에서 정보를 압축해서 저장하기 때문에 정보의 손실이 발생할 수밖에 없고, 이 때문에 특정 입력 조건에 대해서는 사실 여부보다는 지식를 표현하는 국소적인 패턴이 더 큰 영향을 주면서 답변이 생성될 수 있기 때문입니다.
  이러한 문제를 극복하기 위해서는 RAG(Retrieval Augmented Generation) 기술이 필수입니다.
  RAG는 질문에 적합한 레퍼런스 추출을 위해 검색엔진을 활용하고 답변 생성을 위해 LLM(Large Language Model)을 활용합니다.
  이때 LLM은 스스로 알고 있는 지식을 출력하기보다는 언어 추론 능력을 극대화하는 것에 방점을 둡니다.
  이렇게 사실에 기반한 지식 정보를 토대로 질문에 답을 하고 출처 정보도 같이 줄 수 있기 때문에 사용자는 훨씬 더 안심하고 정보를 소비할 수 있게 됩니다.
  아래의 그림이 바로 RAG 구조입니다.  
    
  ![image](https://github.com/UpstageAILab/upstage-ai-final-ir2/blob/main/docs/rag_structure.png)  
  
- RAG는 질문에 적합한 레퍼런스 추출을 위해 검색엔진을 활용하고 답변 생성을 위해 LLM(Large Language Model)을 활용합니다.
  이때 LLM은 스스로 알고 있는 지식을 출력하기보다는 언어 추론 능력을 극대화하는 것에 방점을 둡니다.
  이렇게 사실에 기반한 지식 정보를 토대로 질문에 답을 하고 출처 정보도 같이 줄 수 있기 때문에 사용자는 훨씬 더 안심하고 정보를 소비할 수 있게 됩니다.
- 이번 대회에서는 과학 상식을 질문하는 시나리오를 가정하고 과학 상식 문서 4200여개를 미리 검색엔진에 색인해 둡니다.
  대화 메시지 또는 질문이 들어오면 과학 상식에 대한 질문 의도인지 그렇지 않은 지 판단 후에 과학 상식 질문이라면 검색엔진으로부터 적합한 문서들을 추출하고 이를 기반으로 답변을 생성합니다. 
  만일 과학 상식 이외의 질문이라면 검색엔진을 활용할 필요 없이 적절한 답을 바로 생성합니다.
- 마지막으로, 본 프로젝트는 모델링에 중점을 둔 대회가 아니라 RAG(Retrieval Augmented Generation) 시스템의 개발에 집중하고 있습니다.
  이 대회는 여러 모델과 다양한 기법, 그리고 앙상블을 활용하여 모델의 성능을 향상시키는 일반적인 모델링 대회와는 다릅니다.
  대신에 검색 엔진이 올바른 문서를 색인했는지, 그리고 생성된 답변이 적절한지 직접 확인하는 것이 중요한 대회입니다.
  따라서, 참가자들은 작은 규모의 토이 데이터셋(10개 미만)을 사용하여 초기 실험을 진행한 후에 전체 데이터셋에 대한 평가를 진행하는 것을 권장합니다.
  실제로 RAG 시스템을 구축할 때에도 이러한 방식이 일반적으로 적용되며, 이를 통해 실험을 더욱 효율적으로 진행할 수 있습니다.
  따라서 이번 대회는 2주간 진행되며, 하루에 제출할 수 있는 횟수가 5회로 제한됩니다.

### Timeline

- April 8, 2024 - Start Date: Studying Lectures
- April 11, 2024 - First Mentoring
- April 15, 2024 - Starting Project Date
- April 18, 2024 - Second Mentoring
- April 23, 2024 - Third Mentoring
- May 2, 2024 - Final submission deadline

## 2. Components

### Directory

```
├── code
├── configs
├── datasets
├── logger
├── models
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── data
│   ├── documents.jsonl
│   ├── eval.jsonl
│   ├── generated_questions_single_train.csv
│   ├── generated_questions_single_val.csv
│   ├── generated_questions_single_test.csv
│   ├── generated_questions_multiple_train.csv
│   ├── generated_questions_multiple_val.csv
│   └── generated_questions_multiple_test.csv
│   └── gemini_q_generation_sample.ipynb
└────────── train.py
```

## 3. Data descrption

### Dataset overview

- 과학 상식 문서 4272개
- ko_ai2_arc__ARC_Challenge와 ko_mmlu 데이터
- 총 63개의 데이터 소스 (ko_mmlu__human_sexuality__train, ko_mmlu__human_sexuality__test 등을 별개로 카운트,
  또한 ko_mmlu__human_sexuality__train과 ko_mmlu__conceptual_physics__train 도 별개로 카운트)
- 파일 포맷은 각 line이 json 데이터인 jsonl 파일입니다.

**학습 데이터**  
```
{"docid": "42508ee0-c543-4338-878e-d98c6babee66", "src": "ko_mmlu__nutrition__test", "content": "건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다.
에너지 균형은 에너지 섭취와 에너지 소비의 수학적 동등성을 의미합니다. 일반적으로 건강한 사람은 1-2주의 기간 동안 에너지 균형을 달성합니다. 이 기간 동안에는 올바른 식단과 적절한 운동을 통해 에너지 섭취와 에너지 소비를 조절해야 합니다.
식단은 영양가 있는 식품을 포함하고, 적절한 칼로리를 섭취해야 합니다. 또한, 운동은 에너지 소비를 촉진시키고 근육을 강화시킵니다. 이렇게 에너지 균형을 유지하면 건강을 유지하고 비만이나 영양 실조와 같은 문제를 예방할 수 있습니다.
따라서 건강한 사람은 에너지 균형을 평형 상태로 유지하는 것이 중요하며, 이를 위해 1-2주의 기간 동안 식단과 운동을 조절해야 합니다."}
{"docid": "7a3e9dc2-2572-4954-82b4-1786e9e48f1f", "src": "ko_ai2_arc__ARC_Challenge__test", "content": "산꼭대기에서는 중력이 아주 약간 변합니다.
이는 무게에 영향을 미칩니다. 산꼭대기에서는 무게가 감소할 가능성이 가장 높습니다. 중력은 지구의 질량에 의해 결정되며, 산꼭대기에서는 지구의 질량과의 거리가 더 멀어지기 때문에 중력이 약간 감소합니다.
따라서, 산꼭대기에서는 무게가 더 가볍게 느껴질 수 있습니다."}  
```

**평가 데이터**    
20개의 멀티턴 대화와 20개의 과학 상식 이외의 일상 대화  
```
{"eval_id": 0, "msg": [{"role": "user", "content": "나무의 분류에 대해 조사해 보기 위한 방법은?"}]}
{"eval_id": 1, "msg": [{"role": "user", "content": "각 나라에서의 공교육 지출 현황에 대해 알려줘."}]}
{"eval_id": 2, "msg": [{"role": "user", "content": "기억 상실증 걸리면 너무 무섭겠다."}, {"role": "assistant", "content": "네 맞습니다."}, {"role": "user", "content": "어떤 원인 때문에 발생하는지 궁금해."}]}
{"eval_id": 3, "msg": [{"role": "user", "content": "통학 버스의 가치에 대해 말해줘."}]}
{"eval_id": 4, "msg": [{"role": "user", "content": "Dmitri Ivanovsky가 누구야?"}]}
{"eval_id": 36, "msg": [{"role": "user", "content": "니가 대답을 잘해줘서 너무 신나!"}]}
```

**평가 방법**    
mAP (mean Average Precision)  




### EDA

- 과학 질문의 종류
src에서 출처인 ko_mmlu와 ko_ai2를 제외하고, 뒷부분의 train, validation, test를 제외한 가운데 부분을 과학 지식 domain으로 설정한다.
ko_mmlu__conceptual_physics__test
ko_ai2_arc__ARC_Challenge__test
위 두 예시인 경우 conceptual_physics와 ARC_Challenge로 만들어서 domain에 할당한다.
모든 domain을 모아보면 아래와 같다.  

```
'nutrition', 'conceptual_physics', 'ARC_Challenge',
'human_sexuality', 'virology', 'human_aging',
'high_school_biology', 'high_school_physics', 'college_biology',
'computer_security', 'anatomy', 'college_physics',
'medical_genetics', 'electrical_engineering', 'college_medicine',
'college_chemistry', 'astronomy', 'college_computer_science',
'global_facts', 'high_school_chemistry',
'high_school_computer_science'
```

### Data Processing

- 현재 documents.jsonl에는 정답에 해당하는 문서만 존재하고 질문(혹은 질의)가 없다.
- 주어진 문서에 대한 질의가 없으므로 다른 LLM을 활용하여 생성.
- 무료 API인 Google의 Gemini를 이용해 생성.
- 멘토님의 조언에 따라서 질의와 응답의 pair를 Cosine Embedding Loss로 최적화 시도.
- 이를 위해서는 질의와 응답 페어들을, positive pair와 negative pair들로 만들어야한다.
- Positive pair는 관련이 있는 질의와 응답 쌍이다.
- Negative pair는 관련이 없는 질의와 응답 쌍이다.

e.g. 
질의 Q가 "태양의 지름이 얼마야?" 일 때,

관련이 있는 문서는 
"태양은 지구 지름의 109배인 139만㎞, 무게는 지구보다 무려 33만 2,900배나 무겁습니다.
태양계 전체 질량의 99.8% 이상을 차지하고, 태양계의 중심에 위치하여 지구를 포함한 8개 행성과 위성, 혜성 등의 운동을 지배하고 있는 별입니다." 이고,

관련이 없는 문서는
"인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다."다.

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## Retrospective
- 멘토님이 질의와 응답에 대한 hard negative pairs를 생성하라고 조언해주셨다.
  이는 똑같은 물리학이라도 전자기학과 양자역학이 서로 다르기 때문에,
  커다랗게는 같은 도메인일지라도 세부 내용을 달리하여 보다 섬세하게 학습을 할 수 있도록 한다.
  하지만 이는 사람이 수작업으로 매칭을 해야하고 도메인 지식이 많이 요구되기 때문에 수행할 수 없었다.

### Meeting Log

- [IR 멘토링 기록](https://docs.google.com/document/d/1BLMt-JVnX6Ira2RJkNDoZXio77PSNSD8WyvyHhn2sAA/)

### Reference

- [태양의 지름](https://www.science.go.kr/board?menuId=MENU00366&siteId=)
- [인공지능이란 무엇인가](https://cloud.google.com/learn/what-is-artificial-intelligence?hl=ko)
