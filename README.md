# PyTorch Lecture Notebooks

**웹 교재:** [PyTorch 딥러닝 코드 해설서](docs/index.html) · [LLM 파인튜닝 심화교안](docs/llm-finetuning/index.html)

PyTorch(파이토치) 핵심 개념을 Google Colab에서 실행하며 이해합니다.
아래 "출처 및 참고자료" 등을 활용하여 인공지능 이해를 높이고,
아울러, 최신동향과 코드 실습을 병행하여 실력을 키웁니다.

## 학습 흐름

1. 개념 설명을 읽습니다.
2. 코드 셀을 위에서 아래로 실행합니다.
3. 출력 결과와 해설을 비교합니다.
4. 미니 실무 예제와 확인 문제로 복습합니다.

## 실행 환경

- Python(파이썬) 3
- PyTorch(파이토치)
- NumPy(넘파이, 수치 계산 라이브러리)
- 권장 환경: Google Colab
- CPU(시피유, 중앙처리장치)만으로 전체 실습 가능

## 저장소 구성

다음 구조로 파일을 배치합니다.

```text
pytorch_Learning/
├── README.md
├── .gitignore
└── PyTorch/
    └── PyTorch_*****.ipynb
```

## 자료 출처 및 참고자료
- [PyTorch 공식 Tensor 튜토리얼](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- 성킴(Sung Kim, 성 킴) 교수 — [모두를 위한 머신러닝/딥러닝 강의 홈페이지](https://hunkim.github.io/ml/)
- 박성호 교수 — [NeoWizard GitHub PyTorch 2.x 실습 자료](https://github.com/neowizard2018/neowizard/tree/master/PyTorch)
- Andrew Ng(앤드루 응) 교수 — [공식 홈페이지](https://www.andrewng.org/)
- Andrew Ng(앤드루 응) 교수 강좌 - (https://www.coursera.org/specializations/machine-learning-introduction)
- 박해선 — 한빛미디어 [《혼자 공부하는 머신러닝+딥러닝(개정판)》](https://github.com/rickiepark/hg-mldl2))
- 사이토고키/개앞맵시 — 한빛미디어 [《밑바닥부터 시작하는 딥러닝》](https://github.com/youbeebee/deeplearning_from_scratch)

## 📘 교육용 HTML 공통 제작 규칙

이 저장소의 교육자료(웹페이지·강의 HTML)는 [Yuul-Jung 교육용 HTML 공통 표준](https://github.com/Yuul-Jung/rule_md/tree/main/web/education-html)을 적용합니다.

- [제작 규칙](https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md)
- [다른 저장소 적용 안내](https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/INTEGRATION.md)
- [기준 디자인 사례](https://github.com/Yuul-Jung/ai-agent-engineering)

공통 책장 이미지, 컴팩트한 상단 배너, 좌측 전체 장/절 탐색, 모바일 메뉴, 파일 버전 및 검증 기준을 따릅니다.

```bash
# 사이트 다시 만들기 (Ubuntu)
python3 -m venv .venv && source .venv/bin/activate   # 가상환경 준비
python -m pip install mistune                        # Markdown 변환기 설치
python build.py                                      # CNN 코드 해설서 → docs/
python build.py course_llm.json                      # LLM 파인튜닝 교안 → docs/llm-finetuning/
```
