#!/usr/bin/env python3
"""Generate beginner-friendly Python comments, folder explanations, and Korean TTS MP3 files."""
from __future__ import annotations

import argparse
import ast
import asyncio
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICE_NAME = "ko-KR-SunHiNeural"  # Korean female voice


def find_python_files() -> list[Path]:
    files = sorted(ROOT.glob("chapters/chapter*/practice.py"))
    backend = ROOT / "backend/app/main.py"
    if backend.exists():
        files.append(backend)
    return files


def line_explain(stripped: str) -> str:
    if stripped.startswith("from ") or stripped.startswith("import "):
        return "필요한 도구를 가져와요."
    if stripped.startswith("def "):
        name = stripped[4:].split("(", 1)[0].strip()
        return f"`{name}` 함수를 만들어요."
    if stripped.startswith("class "):
        name = stripped[6:].split("(", 1)[0].split(":", 1)[0].strip()
        return f"`{name}` 클래스를 만들어요."
    if stripped.startswith("@"):
        return "다음 함수에 특별한 설정(데코레이터)을 붙여요."
    if stripped.startswith("if "):
        return "조건이 맞는지 확인해요."
    if stripped.startswith("elif "):
        return "앞 조건이 아니면 다른 조건을 확인해요."
    if stripped.startswith("else"):
        return "조건이 모두 아니면 이 부분을 실행해요."
    if stripped.startswith("for "):
        return "같은 동작을 여러 번 반복해요."
    if stripped.startswith("while "):
        return "조건이 맞는 동안 반복해요."
    if stripped.startswith("return "):
        return "함수 결과를 돌려줘요."
    if stripped.startswith("try:"):
        return "오류가 날 수 있는 동작을 시도해요."
    if stripped.startswith("except "):
        return "오류가 나면 안전하게 처리해요."
    if stripped.startswith("with "):
        return "파일/자원을 안전하게 열고 닫아요."
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return "이 파일 설명(문서 문자열)을 적어요."
    if "=" in stripped:
        return "값을 저장하거나 바꿔요."
    return "다음 코드를 실행해요."


def add_line_comments(py_file: Path) -> bool:
    text = py_file.read_text(encoding="utf-8")
    marker = "# [초등학생 설명 주석 적용됨]"
    if marker in text:
        return False

    out: list[str] = [marker + "\n"]
    for raw in text.splitlines(keepends=True):
        stripped = raw.strip()
        indent = raw[: len(raw) - len(raw.lstrip(" "))]

        if stripped == "" or stripped.startswith("#"):
            out.append(raw)
            continue

        explain = line_explain(stripped)
        out.append(f"{indent}# 설명: {explain}\n")
        out.append(raw)

    py_file.write_text("".join(out), encoding="utf-8")
    return True


def summarize_python_file(py_file: Path) -> tuple[str, list[str], list[str]]:
    text = py_file.read_text(encoding="utf-8")
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or "이 파일은 Python 실습 예제를 담고 있어요."
    funcs: list[str] = []
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                imports.append(base)

    doc_line = doc.strip().splitlines()[0]
    return doc_line, sorted(set(funcs)), sorted(set(imports))


def make_explain_md(folder: Path) -> None:
    py_files = sorted(folder.glob("*.py"))
    title = f"# Python 쉬운 설명 ({folder.name})\n\n"
    intro = (
        "이 문서는 초등학생도 따라할 수 있도록, 폴더 안 Python 파일을 아주 쉽게 설명해요.\n\n"
        "## 먼저 기억해요\n"
        "- `python 파일이름.py` 로 실행해요.\n"
        "- 에러가 나면 메시지를 읽고 천천히 한 줄씩 확인해요.\n"
        "- `run()` 함수가 있으면 그 함수가 핵심 실습이에요.\n\n"
        "## 파일별 설명\n"
    )

    sections: list[str] = []
    for py in py_files:
        doc_line, funcs, imports = summarize_python_file(py)
        funcs_line = ", ".join(f"`{f}`" for f in funcs) if funcs else "없음"
        imports_line = ", ".join(f"`{i}`" for i in imports[:8]) if imports else "없음"
        sections.append(
            f"\n### {py.name}\n"
            f"- 이 파일은: {doc_line}\n"
            f"- 중요한 함수: {funcs_line}\n"
            f"- 사용하는 도구: {imports_line}\n"
            f"- 직접 해보기: `python {py.name}` 실행 후 결과를 읽어보세요.\n"
        )

    outro = (
        "\n## 한 줄 정리\n"
        "이 폴더의 코드는 데이터를 읽고, 계산하고, 결과를 보여주는 연습이에요.\n"
    )

    content = title + intro + "".join(sections) + outro
    (folder / "python_explain.md").write_text(content, encoding="utf-8")


def markdown_to_speech_text(md_text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", md_text)
    text = re.sub(r"#+\s*", "", text)
    text = text.replace("-", "")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


async def write_mp3(md_file: Path) -> None:
    try:
        import edge_tts  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("edge-tts is not installed. Run: pip install edge-tts") from exc

    text = markdown_to_speech_text(md_file.read_text(encoding="utf-8"))
    out = md_file.with_name("python_explain_ko_female.mp3")
    communicate = edge_tts.Communicate(text=text, voice=VOICE_NAME, rate="-8%")
    await communicate.save(str(out))


async def write_all_mp3(md_files: list[Path]) -> None:
    for md in md_files:
        await write_mp3(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["comments", "explain", "audio", "all"],
        default="all",
        help="Select processing mode.",
    )
    args = parser.parse_args()

    py_files = find_python_files()
    folders = sorted({py.parent for py in py_files})

    if args.mode in {"comments", "all"}:
        changed = 0
        for py in py_files:
            if add_line_comments(py):
                changed += 1
        print(f"[comments] updated files: {changed}")

    if args.mode in {"explain", "all"}:
        for folder in folders:
            make_explain_md(folder)
        print(f"[explain] generated markdown files: {len(folders)}")

    if args.mode in {"audio", "all"}:
        md_files = [folder / "python_explain.md" for folder in folders if (folder / "python_explain.md").exists()]
        asyncio.run(write_all_mp3(md_files))
        print(f"[audio] generated mp3 files: {len(md_files)}")


if __name__ == "__main__":
    main()
