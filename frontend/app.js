const listElement = document.getElementById("chapter-list");
const resultElement = document.getElementById("result");

async function loadChapters() {
  const res = await fetch("/api/chapters");
  const chapters = await res.json();

  listElement.innerHTML = "";
  chapters.forEach((chapter) => {
    const li = document.createElement("li");
    li.className = "chapter-item";

    const title = document.createElement("span");
    title.textContent = `${chapter.id} · ${chapter.title}`;

    const button = document.createElement("button");
    button.textContent = "실행";
    button.onclick = async () => runChapter(chapter.id);

    li.appendChild(title);
    li.appendChild(button);
    listElement.appendChild(li);
  });
}

async function runChapter(chapterId) {
  resultElement.textContent = "실행 중...";
  const res = await fetch(`/api/chapters/${chapterId}/run`, { method: "POST" });
  const data = await res.json();
  resultElement.textContent = JSON.stringify(data, null, 2);
}

loadChapters();
