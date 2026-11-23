/*
 * 교육용 HTML 공통 동작 — 유지보수용 원본 (STANDARD 4·10.3·10.4)
 * 페이지 본문은 바꾸지 않고 탐색·이전/다음·모바일 메뉴·코드 복사만 덧붙인다.
 * 입력: 페이지의 window.COURSE (tools/guide/build_guides.py가 실제 존재하는 장만 넣음),
 *       body[data-chapter], main section[id] > h2.
 * 이관 원본과 구문 트리(acorn AST)가 같음을 확인함.
 */
(() => {
  "use strict";
  const data = window.COURSE || { title: "교육과정", chapters: [] };
  const nav = document.getElementById("course-nav");
  if (!nav) return;
  const current = document.body.dataset.chapter || "";
  const pageNames = data.chapters || [];
  // 1. 좌측 탐색 (STANDARD 4.1 — 기준: ai-agent-engineering)
  //    과정 머리글 → 전체 학습 안내서(INDEX) → 접이식 '전체 장 바로가기' → 현재 쪽 세부 목차
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const numText = (n) => String(n).padStart(2, "0");
  const mainCount = pageNames.filter((ch) => !String(ch.number).includes(".")).length;
  const header = el("div", "course-nav-header");
  header.append(el("span", "", data.short_title || data.title), el("span", "", mainCount + " CHAPTERS"));
  nav.append(header);
  const home = el("a", "course-nav-home", "⌂  " + (data.home_label || "전체 학습 안내서"));
  home.href = "index.html";
  if (!current || !pageNames.some((ch) => ch.id === current)) home.setAttribute("aria-current", "page");
  nav.append(home);
  const box = el("details", "course-chapters");
  box.open = true;
  box.append(el("summary", "", "전체 장 바로가기"));
  const chLinks = el("div", "course-ch-links");
  let currentChapter = null;
  // 보조 장(07.1·12.3 등)은 같은 본 장을 보고 있을 때만 펼쳐 목록을 짧게 유지
  const baseOf = (n) => String(n).split(".")[0];
  const curBase = baseOf(pageNames.find((ch) => ch.id === current)?.number ?? "");
  pageNames.forEach((ch) => {
    if (String(ch.number).includes(".") && baseOf(ch.number) !== curBase) return;
    const a = el("a", "course-ch-link");
    a.href = ch.file;
    a.append(el("span", "course-ch-num", numText(ch.number)), el("span", "", ch.title));
    if (ch.id === current) {
      a.setAttribute("aria-current", "page");
      currentChapter = ch;
    }
    chLinks.append(a);
  });
  box.append(chLinks);
  nav.append(box);

  // 2. 현재 쪽 세부 목차: 절 제목(h2)을 번호·제목으로 나눠 표시하고, 화면에 보이는 절을 강조
  const sections = [...document.querySelectorAll("main section[id]")];
  if (sections.length) {
    nav.append(el("div", "course-toc-heading", currentChapter ? "제" + numText(currentChapter.number) + "장 세부 목차" : "INDEX 목차"));
    const sub = el("ol", "toc-sections");
    sections.forEach((sec) => {
      const li = el("li"),
        a = el("a");
      const h2 = sec.querySelector("h2");
      const num = h2?.querySelector(".num")?.textContent.trim() || "";
      let text = h2?.textContent.trim().replace(/\s+/g, " ") || sec.id;
      if (num && text.startsWith(num)) text = text.slice(num.length).trim();
      a.href = "#" + sec.id;
      a.append(el("span", "n", num || "·"), el("span", "", text));
      li.append(a);
      sub.append(li);
    });
    nav.append(sub);
    const items = [...sub.querySelectorAll("a")];
    const mark = (id) =>
      items.forEach((x) => {
        const on = x.getAttribute("href") === "#" + id;
        x.classList.toggle("active", on);
        if (on) x.setAttribute("aria-current", "location");
        else x.removeAttribute("aria-current");
      });
    let ticking = false;
    const sync = () => {
      ticking = false;
      const limit = Math.min(180, innerHeight * 0.28);
      let best = sections[0];
      sections.forEach((x) => {
        if (x.getBoundingClientRect().top <= limit) best = x;
      });
      mark(best.id);
    };
    addEventListener("scroll", () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(sync);
      }
    }, { passive: true });
    sync();
  }

  // 3. 본문 끝 이전/다음 장 링크 (처음·마지막 장은 전체 목차로)
  const idx = pageNames.findIndex((ch) => ch.id === current);
  const links = document.getElementById("chapter-links");
  if (links && idx >= 0) {
    const prev = pageNames[idx - 1],
      next = pageNames[idx + 1];
    const add = (ch, label) => {
      let a = document.createElement("a");
      a.href = ch?.file || "index.html";
      a.textContent = label + (ch ? " · " + ch.title : " · 전체 목차");
      links.append(a);
    };
    add(prev, "← 이전");
    add(next, "다음 →");
  }

  // 4. 모바일 서랍 메뉴: 버튼·덮개 클릭, 링크 선택, Escape 키로 닫기
  const toggle = document.getElementById("mobile-menu"),
    shade = document.getElementById("shade");
  const close = () => {
    nav.classList.remove("open");
    shade?.classList.remove("open");
    toggle?.setAttribute("aria-expanded", "false");
  };
  toggle?.addEventListener("click", () => {
    let open = !nav.classList.contains("open");
    nav.classList.toggle("open", open);
    shade?.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
  });
  shade?.addEventListener("click", close);
  nav.addEventListener("click", (e) => {
    if (e.target.closest("a")) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });

  // 5. 코드 박스 복사 버튼: pre의 소스 텍스트만 복사 (제목 줄 제외)
  document.querySelectorAll(".codeblock").forEach((block) => {
    const pre = block.querySelector("pre");
    if (!pre) return;
    const bar = block.querySelector(".codebar");
    if (!bar) return;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = "복사";
    b.setAttribute("aria-label", "코드 복사");
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.textContent);
        b.textContent = "복사 완료";
        setTimeout(() => (b.textContent = "복사"), 1200);
      } catch {
        b.textContent = "복사 실패";
      }
    });
    bar.append(b);
  });
})();
