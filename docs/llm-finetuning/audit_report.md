# LLM 파인튜닝 12차시 적대적 검증 및 개선 보고서

검증·개선 기준일: **2026-11-05**  
원본: 사용자가 제공한 12차시 과정 안내서와 기존 `LLM_파인튜닝_12차시_통합본.zip`  
게시 저장소: `Yuul-Jung/Pytorch_Learning` / `docs/llm-finetuning/`  
최종 소스 커밋: [`c192ec55eec840a5951f4d5d275a59b7d6df110c`](https://github.com/Yuul-Jung/Pytorch_Learning/commit/c192ec55eec840a5951f4d5d275a59b7d6df110c)

## 검증 범위와 결과

| 검사 | 결과 | 제한 |
|---|---|---|
| 12차시 Markdown·PDF 존재 | PASS | 전체 차시 파일 확인 |
| PDF 분량·한글 | PASS | **총 127페이지, 차시별 10~11페이지**, 모든 PDF 페이지에서 한글 텍스트 확인 |
| Python 소스 정적 구문 | PASS | 16개 Python 파일 AST 구문 검사, 실제 라이브러리·GPU 실행과 다름 |
| 학습/평가 데이터 | PASS | 07차시 원본 데이터 → 전처리 결과 10/2건, 질문 기준 train/eval 중복 0 |
| 비GPU 실행 검사 | PASS | 07 전처리 실실행, 08~10 dry-run, 11·12 단위 테스트 |
| 공개 사이트 상대링크 | PASS | 목차·12차시·검증 보고·내부 코드/자료 링크 연결 점검 |
| 공개 자료 민감정보 검사 | PASS | 토큰·이메일·절대 개인 경로 패턴 미발견 (기계 검사 범위) |
| 실제 GPU 학습/성능 향상 | **미검증** | CUDA/NVIDIA GPU 기반 학습을 수행하지 않음 |
| GGUF·Ollama 실물 실행 | **미검증** | 병합·변환·등록·추론 end-to-end 수행하지 않음 |
| GitHub 저장소 main 반영 | PASS | GitHub API에서 최종 커밋과 게시 파일 확인 |
| GitHub Pages 실서비스 HTTP | **미확인** | Pages 배포 응답을 직접 확인하지 못함. GitHub Pages 설정에서 `main /docs` 확인 필요 |

## 결함 및 조치

1. **P1 — 07차시 전처리 원본과 출력물 누락:** `lesson07_raw.jsonl` 및 `processed/train.jsonl`, `processed/validation.jsonl` 복원. 08~10차시 학습/검증 데이터 연결을 10/2건으로 통일. 기존 09차시 축소 샘플 4/1건은 별도 예제로 유지.
2. **P1 — 01차시 실습 코드 파일 누락:** 안내서에서 실행하도록 명시된 `check_env.py`, `report_versions.sh`를 복원. 명령어·출력 메모리 단위를 GiB로 수정하고 PDF/HTML 재동기화.
3. **P1 — 실습 코드 접근성:** 12개 HTML 페이지에 동봉된 `.py`/`.sh` 실습 코드 링크를 추가하고 GitHub 파일 존재 검증.
4. **P1 — 채팅 템플릿 특수 토큰 중복 가능성:** 03·06차시에서 `apply_chat_template(tokenize=True)`를 직접 사용하도록 정리.
5. **P1 — 학습 경로 및 라이브러리 API 불일치:** 11차시 어댑터 기본 경로를 09차시 저장 경로에 맞추고, 10차시 TRL의 `quantization_config`/`peft_config`를 공식 API와 대조해 수정.
6. **P2 — 버전 및 평가 정확성:** `dtype` 사용 기준 정리, 12차시 GPU 시간 측정 이전 CUDA 동기화 추가.
7. **P2 — GGUF 배포 정확성:** 병합, F16 GGUF 변환, GGUF 양자화, Ollama 등록을 별도 검증 게이트로 분리해 설명.
8. **P2 — 문서 내용 밀도:** 11차시 설명과 실습단계를 확장하고, 개선 Markdown 기반으로 12개 PDF를 전부 재생성.

## 교육 자료 사용 시 주의

- 60분 교육 편성은 **교육과정 시간표 기준**이며, 실제 PC/GPU 구성과 다운로드 속도에 따라 끝나는 시간은 달라질 수 있습니다.
- `Qwen/Qwen2.5-0.5B-Instruct`는 교안 제작상 선택한 예제 모델입니다.
- dry-run과 구문 검사는 모델 가중치 다운로드, 4비트 학습, CUDA 메모리 사용량, 품질 향상 검증의 증거가 아닙니다.
- 교육에 투입하기 전 CUDA 드라이버/PyTorch/Transformers/TRL/PEFT/bitsandbytes 버전을 기록하고 **09~12차시 관통 실험**을 실제 GPU에서 별도로 수행해야 합니다.
- GitHub Pages 배포 후 공개 URL에서 메인 목차, 차시 이동, 코드, PDF를 브라우저로 확인해야 합니다.

## 공식 참고문서 (2026-11-05 대조)

- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL PEFT: https://huggingface.co/docs/trl/peft_integration
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- Transformers 채팅 템플릿: https://huggingface.co/docs/transformers/main/chat_templating
- Transformers bitsandbytes 양자화: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Ollama 모델 가져오기: https://docs.ollama.com/import