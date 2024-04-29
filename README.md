# Scientific Knowledge Question Answering
## IR 2조

| ![김태한](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김소현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김준호](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최장원](https://avatars.githubusercontent.com/u/156163982?v=4) |  
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |  
|   [김태한](https://github.com/UpstageAILab)   |    [김소현](https://github.com/UpstageAILab)       |   [김준호](https://github.com/UpstageAILab)     |            [최장원](https://github.com/UpstageAILab)             |   
|    팀장, 리서치, 데이터생성, 모델링   |      리서치, 데이터생성, 모델링          |           리서치, 데이터생성, 모델링         |     리서치, 데이터생성, 모델링    |  

## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

- 과학 지식 질의 응답 시스템 구축
- LLM은 최근 좋은 성능을 보이지만 Hallucination과 Knolwedge Cut-off 현상이 있다.
- 위 단점을 극복하기 위해서 RAG (Retrieval Augmented Generation)를 도입한다.
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

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- 과학 상식 문서 4272개
- ko_ai2_arc__ARC_Challenge와 ko_mmlu 데이터
- 총 63개의 데이터 소스 (ko_mmlu__human_sexuality__train, ko_mmlu__human_sexuality__test 등을 별개로 카운트,
  또한 ko_mmlu__human_sexuality__train과 ko_mmlu__conceptual_physics__train 도 별개로 카운트)	

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- 멘토님의 조언에 따라서 질의와 응답의 pair를 Cosine Embedding Loss로 최적화 시도.
- 이를 위해서는 질의와 응답 페어들을, positive pair와 negative pair를로 만들어야한다.
- 주어진 문서에 대한 질의가 없으므로 다른 LLM을 활용하여 생성.
- 무료 API인 Google의 Gemini를 이용해 생성.

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

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
