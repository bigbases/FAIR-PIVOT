# Issue-based evaluation 

## 1. 이 평가가 말하는 것

정치·사회 정책 질문 하나에 대해, 모델이 짧은 문단 형태로 입장을 말하게 한다.  
설문에서 숫자만 고르게 하는 방식이 아니라, 생성된 텍스트가 좌우 스펙트럼에서 어디에 가깝게 읽히는지를 보는 쪽에 가깝다.

같은 질문이라도 프롬프트 조건(예: 좌·우·혼합·균형 등)과 백본 모델을 바꿔 가며 여러 번 돌릴 수 있다.  
선택적으로, 같은 모델이 자기 답에 대해 1~5 점수를 스스로 매기는 self-report도 같은 배치 안에서 뽑을 수 있다.

그 다음 단계에서 외부 API judge(예: 서로 다른 두 제공자)가 응답만 보고 같은 1~5 척도로 채점한다.  
self와 external을 나란히 두면, “모델이 스스로 보는 위치”와 “밖에서 보는 위치”의 차이를 같이 논의할 수 있다.

---

## 2. 파이프라인을 두 덩어리로 나누기

첫째 덩어리는 로컬 생성이다.  
GPU가 있는 머신에서 Hugging Face 계열 모델(필요하면 LoRA·어댑터)을 올리고, 질문 세트와 조건·백본 조합에 따라 행 단위 JSONL로 저장한다.

둘째 덩어리는 외부 judge이다.  
앞에서 만든 응답 파일을 읽기만 하고, 지정한 API 모델들이 점수와 짧은 근거를 반환한다.  
두 judge를 쓰면 행마다 점수 쌍이 생기고, 이후 단계에서 평균 점수·차이·합의 여부 같은 필드를 붙여 분석용 표 하나(merged)로 모을 수 있다.

순서는 생성이 먼저, judge는 그다음이다.  

---

## 3. 산출물

1. 생성 결과 — (시드, 백본, 조건, 문항) 단위로 한 줄씩: 질문, 응답 텍스트, 메타데이터.
2. 선택 self — 같은 줄에 self 점수·근거 필드가 붙을 수 있다.
3. Judge 원본 — 제공자별로 같은 키(문항·조건 등)에 대응하는 채점 한 줄들.
4. 병합 결과 — 한 문항 단위로 생성 + self + 양쪽 judge를 한 줄로 합친 것.
5. 보내기 — 스프레드시트나 노트북에서 바로 열 수 있는 CSV. 

---

## 4. 요약

issue 평가는, 정책 질문에 대해 모델이 문단형으로 답하게 만든 뒤, 필요하면 같은 모델이 self 점수를 매기고, 그 다음 외부 API judge 두 곳이 같은 답을 1에서 5까지 채점해서 한 줄로 합친다.  

---

## 5. 현재 레포에서 파일 이름

개념은 위와 같고, 이 패키지 안에서는 대략 아래 파일이 그 역할을 한다. 경로는 `implicit_framing_v1/` 기준 상대 경로다.

- `configs/issue_prompts_v1.json` — 질문·anchor 정의
- `scripts/generate_issue_eval.py` — 로컬 생성 진입점
- `scripts/judge_issue_eval.py` — 외부 judge 진입점
- `scripts/_hf_eval_utils.py` — 생성 쪽 공용 (모델·토큰화·저장)
- `scripts/_judge_utils.py` — judge·CSV 공용

선택으로 같이 있으면 분석에 도움이 되는 것들:

- `scripts/analyze_judge_agreement.py`
- `results/tools/plot_issue_judge_condition_means.py`, `plot_issue_judge_condition_means_10run.py`
- `results/tools/plot_issue_judge_per_backbone.py`, `plot_issue_judge_per_backbone_10run.py`

실행 위치는 보통 `scripts/` 디렉터리에서 `python3 generate_issue_eval.py` / `python3 judge_issue_eval.py` 이다. 환경변수 이름은 두 스크립트 상단을 보면 된다.

