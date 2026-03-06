# Vision Captioning & Interactive QA System  
### Florence2 vs BLIP2 on Real-World Low-Quality Images (VizWiz)  

## Project Overview
본 프로젝트는 실제 시각장애인이 촬영한 이미지 DataSet VizWiz를 기반으로,  
최신 SOTA 캡셔닝 모델 Florence2와 BLIP2의 성능을 비교하고,  
생성된 캡션을 활용하여 LLM 기반 질의응답 시스템 및 TTS 음성 출력까지 확장한 멀티모달 파이프라인 시스템이다.  

단순 모델 비교가 아니라,  
실제 사용자가 촬영한 저품질 이미지 환경에서  
"설명 - 질문 - 음성 응답"까지 가능한 구조를 설계하는 것을 목표로 했다.  

---  

## Research Questions  
1. 저품질 실제 이미지 환경에서 Florence2와 BLIP2는 얼마나 안정적으로 설명을 생성하는가?  
2. VizWiz 같은 노이즈 환경에서 생성 문장의 품질은 어느 수준인가?  
3. 캡션을 기반으로 한 LLM QA 시스템은 의미 있는 답변을 생성할 수 있는가?

---  

## System Architecture  
```Plain text
Image
 ↓
Florence2 / BLIP2
 ↓
Generated Caption
 ↓
LLM (Qwen)
 ↓
Text Answer
 ↓
gTTS
 ↓
Speech Output
```
Vision Model = 시각적 이해  
LLM = 언어적 추론  
TTS = 접근성 확장  

---  

## Dataset Understanding (Critical Design Point)  
VizWiz 데이터셋 특징:  
- 실제 시각장애인이 직접 촬영  
- 초점 불안정  
- 조명 문제
- 일부 완전 노이즈 이미지 존재

초기 무작위 랜덤 샘플링 방식으로 진행하려고 하였으나,  
해당 방식이 적합하지 않다고 판단하여 실제 서비스 가능성을 고려한 수동 이미지 선별 전략 사용  

---  

## Evaluation Stategy  
### 사용 지표  
- BELU  
- ROUGE  
- METEOR

### 신뢰성 강화
VizWiz 데이터셋의 GT에는 무의미한 캡션이 존재한다. (완전 노이즈 이미지가 데이터셋 내에 존재)  
> ex) "Quality issues are too severe to recognize visual content."  
이를 필터링하지 않을 경우 Florence와 BLIP2 성능 지표를 구하는 과정에서 점수의 왜곡이 발생하게 됨.  
따라서 아래와 같이 GT 검증 로직을 추가함.
```python
invalid_phrases = [  
    "quality issues are too severe",  
    "unable to recognize",  
    "no way of seeing",  
    "not visible"  
]
```
이를 통해 평가 왜곡 제거 + 정량 지표 신뢰성 확보  

---  

## Experimental Results  
| Metric | Florence2 | BLIP2  |
| ------ | --------- | ------ |
| BLEU   | 0.174     | 0.158  |
| ROUGE  | 0.4636    | 0.4673 |
| METEOR | 0.3876    | 0.3342 |  

- Florence2는 의미 보존 및 문장 정밀도에서 BLIP2에 비해 소폭 우수
- BLIP2는 객체 중심 명사 표현이 강함
- 노이즈 환경에서 두 모델 모두 한계 존재

---  

## Design Decisions
### 1. Florence2 기반 QA 확장 선택 이유
성능 지표 실험 결과 Florence2의 캡셔닝 품질이 더 좋은 지표를 보여주었기 때문에  
QA 파이프라인의 입력으로 Florence2 캡션 사용  

### 2. TTS 확장 이유
VizWiz는 시각장애인 대상 데이터셋이므로  
텍스트 출력만 아니라 음성 출력까지 포함해야 실제 사용 시나리오에 가깝다고 판단.  

---

## QA System Extension  
조금 더 우수한 성능을 보여준 Florence2 모델을 사용하여,  
생성한 캡션을 context로 경량 LLM Qwen과 통합  
- 초기에는 OpenAI API를 사용
- 이후 위 시스템이 추후 시각장애인을 위한 안경 및 선글라스 등등 경량화 및 실사용 확장성을 고려하여 Qwen 모델을 선택

위 프로젝트가 서비스화 된다면 추후 웨어러블 기기 적용 가능성을 고려한 아키텍처를 결정함.  

---  

## Accessibility Enhancement
LLM 응답을 gTTS로 변환하여,  
시각장애인들이 텍스트를 통해 이 서비스를 이용할 수 없음을 청각 출력으로 확장  

---

## Limitations  
- 캡션 품질에 QA 품질 의존
- 추상적 질문에 대한 저품질 응답
- 실시간 최적화 미구현
- 임의 샘플 추출 및 샘플 수 제한


---

## What I Learned
- 멀티모달 모델은 단순 정확도 비교보다 파이프라인 설계 관점이 중요
- Caption 품질이 QA 시스템 전체 성능에 직접적인 영향을 미친다는 점
- Vision과 Language 모델을 분리하여 설계하는 구조가 유지보수 측면에서 유리
- 데이터 특성(VizWiz의 낮은 품질 이미지)이 모델 성능에 큰 영향을 준다는 점

--- 

## Tech Stack  
- PyTorch
- HuggingFace Transformers
- Florence2
- BLIP2 (LAVIS)
- Qwen 1.5-1.8B
- HuggingFace evaluate
- gTTS

---  

## References  
- 
