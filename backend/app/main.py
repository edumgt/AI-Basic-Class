# [초등학생 설명 주석 적용됨]
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from pathlib import Path
# 설명: 필요한 도구를 가져와요.
from typing import Any

# 설명: 필요한 도구를 가져와요.
from fastapi import FastAPI, HTTPException
# 설명: 필요한 도구를 가져와요.
from fastapi.middleware.cors import CORSMiddleware
# 설명: 필요한 도구를 가져와요.
from fastapi.responses import FileResponse
# 설명: 필요한 도구를 가져와요.
from fastapi.staticfiles import StaticFiles
# 설명: 필요한 도구를 가져와요.
from pydantic import BaseModel

# 설명: 값을 저장하거나 바꿔요.
app = FastAPI(title="AI/ML Basic Class API", version="1.0.0")

# 설명: 다음 코드를 실행해요.
app.add_middleware(
    # 설명: 다음 코드를 실행해요.
    CORSMiddleware,
    # 설명: 값을 저장하거나 바꿔요.
    allow_origins=["*"],
    # 설명: 값을 저장하거나 바꿔요.
    allow_credentials=True,
    # 설명: 값을 저장하거나 바꿔요.
    allow_methods=["*"],
    # 설명: 값을 저장하거나 바꿔요.
    allow_headers=["*"],
# 설명: 다음 코드를 실행해요.
)

# 설명: 값을 저장하거나 바꿔요.
BASE_DIR = Path(__file__).resolve().parents[2]
# 설명: 값을 저장하거나 바꿔요.
CHAPTERS_DIR = BASE_DIR / "chapters"
# 설명: 값을 저장하거나 바꿔요.
FRONTEND_DIR = BASE_DIR / "frontend"


# 설명: `ChapterSummary` 클래스를 만들어요.
class ChapterSummary(BaseModel):
    # 설명: 다음 코드를 실행해요.
    id: str
    # 설명: 다음 코드를 실행해요.
    title: str
    # 설명: 다음 코드를 실행해요.
    path: str


# 설명: `ChapterRunResponse` 클래스를 만들어요.
class ChapterRunResponse(BaseModel):
    # 설명: 다음 코드를 실행해요.
    chapter: str
    # 설명: 다음 코드를 실행해요.
    result: dict[str, Any]


# 설명: `ChapterSourceResponse` 클래스를 만들어요.
class ChapterSourceResponse(BaseModel):
    # 설명: 다음 코드를 실행해요.
    chapter: str
    # 설명: 다음 코드를 실행해요.
    source: str


# 설명: `load_chapters` 함수를 만들어요.
def load_chapters() -> list[ChapterSummary]:
    # 설명: 값을 저장하거나 바꿔요.
    items: list[ChapterSummary] = []
    # 설명: 같은 동작을 여러 번 반복해요.
    for chapter_dir in sorted(CHAPTERS_DIR.glob("chapter*")):
        # 설명: 값을 저장하거나 바꿔요.
        readme = chapter_dir / "README.md"
        # 설명: 값을 저장하거나 바꿔요.
        title = chapter_dir.name
        # 설명: 조건이 맞는지 확인해요.
        if readme.exists():
            # 설명: 값을 저장하거나 바꿔요.
            first_line = readme.read_text(encoding="utf-8").splitlines()[0]
            # 설명: 값을 저장하거나 바꿔요.
            title = first_line.replace("#", "").strip()
        # 설명: 다음 코드를 실행해요.
        items.append(
            # 설명: 다음 코드를 실행해요.
            ChapterSummary(
                # 설명: 값을 저장하거나 바꿔요.
                id=chapter_dir.name,
                # 설명: 값을 저장하거나 바꿔요.
                title=title,
                # 설명: 값을 저장하거나 바꿔요.
                path=str(chapter_dir.relative_to(BASE_DIR)),
            # 설명: 다음 코드를 실행해요.
            )
        # 설명: 다음 코드를 실행해요.
        )
    # 설명: 함수 결과를 돌려줘요.
    return items


# 설명: 다음 함수에 특별한 설정(데코레이터)을 붙여요.
@app.get("/api/health")
# 설명: `health` 함수를 만들어요.
def health() -> dict[str, str]:
    # 설명: 함수 결과를 돌려줘요.
    return {"status": "ok"}


# 설명: 다음 함수에 특별한 설정(데코레이터)을 붙여요.
@app.get("/api/chapters", response_model=list[ChapterSummary])
# 설명: `chapters` 함수를 만들어요.
def chapters() -> list[ChapterSummary]:
    # 설명: 함수 결과를 돌려줘요.
    return load_chapters()


# 설명: 다음 함수에 특별한 설정(데코레이터)을 붙여요.
@app.post("/api/chapters/{chapter_id}/run", response_model=ChapterRunResponse)
# 설명: `run_chapter` 함수를 만들어요.
def run_chapter(chapter_id: str) -> ChapterRunResponse:
    # 설명: 값을 저장하거나 바꿔요.
    chapter_path = CHAPTERS_DIR / chapter_id / "practice.py"
    # 설명: 조건이 맞는지 확인해요.
    if not chapter_path.exists():
        # 설명: 값을 저장하거나 바꿔요.
        raise HTTPException(status_code=404, detail="chapter not found")

    # 설명: 값을 저장하거나 바꿔요.
    namespace: dict[str, Any] = {}
    # 설명: 값을 저장하거나 바꿔요.
    code = chapter_path.read_text(encoding="utf-8")
    # 설명: 다음 코드를 실행해요.
    exec(compile(code, str(chapter_path), "exec"), namespace)
    # 설명: 조건이 맞는지 확인해요.
    if "run" not in namespace:
        # 설명: 값을 저장하거나 바꿔요.
        raise HTTPException(status_code=500, detail="run function not found")

    # 설명: 값을 저장하거나 바꿔요.
    result = namespace["run"]()
    # 설명: 함수 결과를 돌려줘요.
    return ChapterRunResponse(chapter=chapter_id, result=result)


# 설명: 다음 함수에 특별한 설정(데코레이터)을 붙여요.
@app.get("/api/chapters/{chapter_id}/source", response_model=ChapterSourceResponse)
# 설명: `chapter_source` 함수를 만들어요.
def chapter_source(chapter_id: str) -> ChapterSourceResponse:
    # 설명: 값을 저장하거나 바꿔요.
    chapter_path = CHAPTERS_DIR / chapter_id / "practice.py"
    # 설명: 조건이 맞는지 확인해요.
    if not chapter_path.exists():
        # 설명: 값을 저장하거나 바꿔요.
        raise HTTPException(status_code=404, detail="chapter not found")

    # 설명: 값을 저장하거나 바꿔요.
    source = chapter_path.read_text(encoding="utf-8")
    # 설명: 함수 결과를 돌려줘요.
    return ChapterSourceResponse(chapter=chapter_id, source=source)


# 설명: 다음 함수에 특별한 설정(데코레이터)을 붙여요.
@app.get("/")
# 설명: `index` 함수를 만들어요.
def index() -> FileResponse:
    # 설명: 함수 결과를 돌려줘요.
    return FileResponse(FRONTEND_DIR / "index.html")


# 설명: 값을 저장하거나 바꿔요.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
