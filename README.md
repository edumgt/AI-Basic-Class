# AI/ML Basic Class 실습 프로젝트

이 저장소의 기존 Markdown 자료(수학/통계 용어, 수식-코드 매핑, Python 예제)를 기반으로, **Chapter01~Chapter30 실습 코드**와 **FastAPI + 프론트엔드 실습 앱**을 구성했습니다.

## 프로젝트 구성

- `chapters/chapter01` ~ `chapters/chapter30`: 챕터별 `README.md` + `practice.py`
- `backend/app/main.py`: FastAPI 백엔드
- `frontend/`: 브라우저에서 챕터 실행/결과 확인 UI
- `requirements.txt`: 실행 의존성

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속 후 챕터를 실행하면 결과(JSON)를 확인할 수 있습니다.

## 학습 흐름

1. chapter01~04: 데이터/수학 기초
2. chapter05~11: 핵심 ML 모델 + 평가/검증
3. chapter12~18: 실무형 전처리/재현성/배포 준비
4. chapter19~20: FastAPI 서빙 및 통합 미니 프로젝트
5. chapter21: 신경망 모델 전체 흐름(요약)
6. chapter22~30: 신경망 학습 요소를 세분화한 확장 실습(행렬/활성화/소프트맥스/손실/역전파/최적화/CNN)
