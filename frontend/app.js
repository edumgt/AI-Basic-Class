const listElement = document.getElementById("chapter-list");
const resultElement = document.getElementById("result");
const sourceElement = document.getElementById("source");
const selectedChapterElement = document.getElementById("selected-chapter");

const offcanvas = document.getElementById("offcanvas");
const overlay = document.getElementById("overlay");
const openMenuBtn = document.getElementById("open-menu");
const closeMenuBtn = document.getElementById("close-menu");

function setMenuOpen(isOpen) {
  offcanvas.classList.toggle("-translate-x-full", !isOpen);
  overlay.classList.toggle("hidden", !isOpen);
  offcanvas.setAttribute("aria-hidden", String(!isOpen));
}

openMenuBtn.addEventListener("click", () => setMenuOpen(true));
closeMenuBtn.addEventListener("click", () => setMenuOpen(false));
overlay.addEventListener("click", () => setMenuOpen(false));

async function loadChapters() {
  const res = await fetch("/api/chapters");
  const chapters = await res.json();

  listElement.innerHTML = "";
  chapters.forEach((chapter) => {
    const li = document.createElement("li");

    const item = document.createElement("div");
    item.className = "chapter-btn";

    const title = document.createElement("button");
    title.className = "chapter-label";
    title.textContent = `${chapter.id} · ${chapter.title}`;
    title.type = "button";
    title.onclick = async () => {
      await showSource(chapter.id, chapter.title);
      setMenuOpen(false);
    };

    const runButton = document.createElement("button");
    runButton.className = "chapter-run";
    runButton.textContent = "실행";
    runButton.type = "button";
    runButton.onclick = async () => {
      await showSource(chapter.id, chapter.title);
      await runChapter(chapter.id);
      setMenuOpen(false);
    };

    item.appendChild(title);
    item.appendChild(runButton);
    li.appendChild(item);
    listElement.appendChild(li);
  });
}

async function showSource(chapterId, chapterTitle) {
  selectedChapterElement.textContent = `${chapterId} · ${chapterTitle}`;
  sourceElement.textContent = "소스 로딩 중...";
  const res = await fetch(`/api/chapters/${chapterId}/source`);
  if (!res.ok) {
    sourceElement.textContent = "소스를 불러오지 못했습니다.";
    return;
  }
  const data = await res.json();
  sourceElement.textContent = data.source;
}

async function runChapter(chapterId) {
  resultElement.textContent = "실행 중...";
  const res = await fetch(`/api/chapters/${chapterId}/run`, { method: "POST" });
  const data = await res.json();
  resultElement.textContent = JSON.stringify(data, null, 2);
}

loadChapters();
