# GitHub 교육용 HTML 작성 규칙

이 저장소에서 새 HTML 교재를 만들거나 기존 교재를 수정할 때는 먼저 다음 문서를 확인한다.
- https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md
- https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/INTEGRATION.md
- 코드: https://github.com/Yuul-Jung/rule_md/blob/main/common/code-style-rules.md
- 도식: https://github.com/Yuul-Jung/rule_md/blob/main/common/flowchart-rules.md

적용 사항:
1. 책장 배너 이미지는 중앙 표준의 hero-bookshelf.jpg와 동일하게 유지한다.
2. 좌측 전체 장 목차와 현재 페이지 절 목차 및 모바일 메뉴를 구현한다.
3. 기존 학습 내용·코드·이미지는 보존하고, 차이가 있는 부분만 수정한다.
4. 수정 시 장별 파일명을 _v01 → _v02로 올리고 이전 생성본은 결과 폴더에서 정리한다.
5. 코드 문법, 내부 링크, 개인정보, 접근성과 브라우저 표시를 검증한다.
6. 확인하지 못한 실행 결과는 미검증으로 표기한다.
7. 원격 문서를 읽을 수 없으면 추정 적용하지 않고 로컬 규칙 사본을 요구한다.

이 저장소 고유 규칙:
- 생성기는 `build.py`, 강좌 구성은 `course.json`, 표준 자산 사본은 `assets/edu/`(site.css·site.js·hero-bookshelf.jpg는 rule_md 원본 그대로, chapter.css만 변환용 추가 스타일)이다.
- 생성 HTML(`index.html`, `chNN_*_vNN.html`)은 직접 고치지 않고 원고를 고친 뒤 `python build.py`로 다시 만든다.
- `course.json`의 `legacy_file`은 이전 주소로, 새 장으로 넘겨 주는 연결 파일만 둔다. 삭제하면 기존 링크가 끊긴다.
- 과정이 둘이다: CNN 코드 해설서(`course.json` → `docs/`)와 LLM 파인튜닝 교안(`course_llm.json` → `docs/llm-finetuning/`). 각각 `python build.py`, `python build.py course_llm.json`으로 만든다.
- 장 원고는 `content/cnn/`, `content/llm/`의 HTML 조각이다. LLM 교안 고유 스타일은 `assets/edu/llm-finetuning.css`(원래 style.css를 본문 범위로 한정한 사본)이다.
- CNN 해설서의 코드·출력은 원본 Notebook의 실제 실행 결과이므로 바꾸지 않는다.
- Notebook 출력에 들어 있던 개인 로컬 경로(`/home/<사용자>/`)는 개인정보 보호를 위해 `~/`로 바꿔 실었다. 다른 출력 값은 바꾸지 않는다.
- 커밋은 하나의 논리적 목적만 담고, 메시지는 한글로 변경 목적을 쓴다.
