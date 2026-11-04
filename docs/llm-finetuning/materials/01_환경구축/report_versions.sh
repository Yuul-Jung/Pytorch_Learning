#!/usr/bin/env bash
# 01차시: Python 가상환경의 의존 패키지를 로그로 기록한다.
set -euo pipefail
mkdir -p docs
python --version | tee docs/python_version.txt
python -m pip freeze > docs/requirements-locked.txt
python check_env.py | tee docs/environment_report.txt
# 주의: 환경 보고에는 장비 정보가 포함되므로 공개 전 확인한다.