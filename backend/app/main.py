from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="AI/ML Basic Class API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
CHAPTERS_DIR = BASE_DIR / "chapters"
FRONTEND_DIR = BASE_DIR / "frontend"


class ChapterSummary(BaseModel):
    id: str
    title: str
    path: str


class ChapterRunResponse(BaseModel):
    chapter: str
    result: dict[str, Any]


class ChapterSourceResponse(BaseModel):
    chapter: str
    source: str


def load_chapters() -> list[ChapterSummary]:
    items: list[ChapterSummary] = []
    for chapter_dir in sorted(CHAPTERS_DIR.glob("chapter*")):
        readme = chapter_dir / "README.md"
        title = chapter_dir.name
        if readme.exists():
            first_line = readme.read_text(encoding="utf-8").splitlines()[0]
            title = first_line.replace("#", "").strip()
        items.append(
            ChapterSummary(
                id=chapter_dir.name,
                title=title,
                path=str(chapter_dir.relative_to(BASE_DIR)),
            )
        )
    return items


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/chapters", response_model=list[ChapterSummary])
def chapters() -> list[ChapterSummary]:
    return load_chapters()


@app.post("/api/chapters/{chapter_id}/run", response_model=ChapterRunResponse)
def run_chapter(chapter_id: str) -> ChapterRunResponse:
    chapter_path = CHAPTERS_DIR / chapter_id / "practice.py"
    if not chapter_path.exists():
        raise HTTPException(status_code=404, detail="chapter not found")

    namespace: dict[str, Any] = {}
    code = chapter_path.read_text(encoding="utf-8")
    exec(compile(code, str(chapter_path), "exec"), namespace)
    if "run" not in namespace:
        raise HTTPException(status_code=500, detail="run function not found")

    result = namespace["run"]()
    return ChapterRunResponse(chapter=chapter_id, result=result)


@app.get("/api/chapters/{chapter_id}/source", response_model=ChapterSourceResponse)
def chapter_source(chapter_id: str) -> ChapterSourceResponse:
    chapter_path = CHAPTERS_DIR / chapter_id / "practice.py"
    if not chapter_path.exists():
        raise HTTPException(status_code=404, detail="chapter not found")

    source = chapter_path.read_text(encoding="utf-8")
    return ChapterSourceResponse(chapter=chapter_id, source=source)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
