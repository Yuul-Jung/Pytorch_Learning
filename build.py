#!/usr/bin/env python3
"""rule_md 교육용 HTML 표준(web/education-html) 생성기 — 기존 교재 변환용 범용판.

입력: course.json + 장별 원고(Markdown .md 또는 '## 제목'으로 절을 나눈 HTML 조각 .html)
출력: course.json의 "out" 폴더(기본: 저장소 루트)에 index.html, chNN_주제_vNN.html
      + 이전 주소(legacy_file)로 들어온 방문자를 새 장으로 넘기는 연결 파일
이미지·스타일·스크립트는 생성 HTML에 내장하고, 본문 그림은 저장소 안 원래 위치를 상대 경로로 참조한다.

실행(Ubuntu):
  python3 -m venv .venv && source .venv/bin/activate   # 가상환경 준비
  python -m pip install mistune                        # Markdown 변환기 설치
  python build.py                                      # 사이트 생성 (course.json)
  python build.py other_course.json                    # 같은 저장소의 다른 과정
기준 표준: https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md
"""
import base64
import html
import json
import os
import re
import urllib.parse
from pathlib import Path

import sys

import mistune

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'course.json')  # 과정이 여럿이면: python build.py 다른과정.json
CONFIG = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
OUT = (ROOT / CONFIG.get('out', '.')).resolve()
ASSETS = ROOT / CONFIG.get('assets', 'assets/edu')
IMAGE = base64.b64encode((ASSETS / 'hero-bookshelf.jpg').read_bytes()).decode('ascii')
CSS = (ASSETS / 'site.css').read_text(encoding='utf-8')
CSS = re.sub(r"url\((['\"])hero-bookshelf\.jpg\1\)", f"url('data:image/jpeg;base64,{IMAGE}')", CSS)
CSS += '\n' + (ASSETS / 'chapter.css').read_text(encoding='utf-8')
for extra in CONFIG.get('extra_css', []):  # 과정 고유 스타일(원본 교재 스타일을 .sec-body 범위로 한정한 사본)
    CSS += '\n' + (ROOT / extra).read_text(encoding='utf-8')
SCRIPT = (ASSETS / 'site.js').read_text(encoding='utf-8')
RULE_URL = 'https://github.com/Yuul-Jung/rule_md/tree/main/web/education-html'
SKIP = set(CONFIG.get('skip_sections', ['목차', '차례', 'Contents', 'Table of Contents']))
MARKDOWN = mistune.create_markdown(escape=False, plugins=['table', 'strikethrough', 'footnotes'])


def escape(s):
    return html.escape(str(s), quote=True)


CHAPTERS = []
for raw in CONFIG['chapters']:
    ch = dict(raw)
    ch['file'] = f"{ch['slug']}_{ch['version']}.html"
    ch.setdefault('nav_title', ch['title'])
    CHAPTERS.append(ch)
BY_NUM = {ch['number']: ch for ch in CHAPTERS}
# 원고·이전 HTML 경로 → 새 장 파일 (장 사이 링크 보정용)
LINK_MAP = {}
for ch in CHAPTERS:
    for key in [ch['source']] + ch.get('aliases', []) + ([ch['legacy_file']] if ch.get('legacy_file') else []):
        LINK_MAP[os.path.normpath(key)] = ch['file']


def label(num):
    """장 번호 표시: 7 → '07', 7.1 → '07.1'."""
    if isinstance(num, float) and not num.is_integer():
        whole, frac = str(num).split('.')
        return f'{int(whole):02d}.{frac}'
    return f'{int(num):02d}'


def fix_links(fragment, src_dir):
    """원고 위치 기준 상대 경로를 출력 폴더 기준으로 바꾸고, 다른 장 원고 링크는 새 장 파일로 바꾼다."""
    def repl(m):
        attr, quote, url = m.group(1), m.group(2), m.group(3)
        if re.match(r'^(?:[a-z][a-z0-9+.-]*:|#|/|data:)', url, re.I):
            return m.group(0)
        path, _, frag = url.partition('#')
        if not path:
            return m.group(0)
        target = os.path.normpath(os.path.join(src_dir, path))
        rel_root = os.path.relpath(target, ROOT)
        if rel_root in LINK_MAP:
            new = LINK_MAP[rel_root]
            new = os.path.relpath(OUT / new, OUT)
            frag = ''  # 이전 절 앵커는 새 번호 체계와 다르므로 장 처음으로 보낸다
        else:
            new = os.path.relpath(target, OUT)
        new = new.replace(os.sep, '/')
        return f'{attr}={quote}{new}{"#" + frag if frag else ""}{quote}'
    return re.sub(r'\b(src|href)=(["\'])([^"\']+)\2', repl, fragment)


def code_boxes(fragment):
    """```언어 코드 블록 → 표준 코드 상자(언어 표시 + 복사 버튼 자리)."""
    def box(m):
        lang = m.group(1) or 'text'
        return (f'<div class="codeblock"><div class="codebar"><span>{escape(lang.upper())}</span></div>'
                f'<pre><code class="language-{escape(lang)}">{m.group(2)}</code></pre></div>')
    out = re.sub(r'<pre><code(?: class="language-([^"]+)")?>(.*?)</code></pre>', box, fragment, flags=re.S)
    # 넓은 표는 가로 스크롤 상자로 감싼다
    return re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>', out, flags=re.S)


def strip_number(title):
    """'3. 제목' / '3장 제목' / '[3] 제목' 같은 원고 번호를 떼어 절 번호(01, 02…)와 겹치지 않게 한다."""
    return re.sub(r'^(?:\d+(?:\.\d+)*[.)]?(?:\s*[|·:—-])?|\[\d+\]|제?\d+절)\s+', '', title).strip() or title


def slug_variants(title):
    """원고 안 링크가 가리키던 옛 제목 앵커(GitHub·pandoc 방식) 후보."""
    plain = html.unescape(re.sub(r'<[^>]+>', '', title)).strip()
    out = set()
    for s in (plain, strip_number(plain)):
        g = re.sub(r'[^\w\- ]', '', s.lower()).replace(' ', '-')
        out |= {g, g.strip('-'), re.sub(r'^[\d\-]+', '', g)}
    return {x for x in out if x}


def nav_only(body_html):
    """이전/다음·목차 링크만 있는 절(표준 탐색과 중복)인지."""
    if '<a ' not in body_html:
        return False
    rest = re.sub(r'<a\b.*?</a>', '', body_html, flags=re.S)
    rest = re.sub(r'<[^>]+>|[\s·|/\-—]', '', rest)
    return len(rest) < 6


def drop_local_toc(fragment):
    """항목이 모두 같은 쪽 앵커 링크뿐인 목록(원고 자체 목차)은 좌측 탐색과 중복이므로 뺀다."""
    def repl(m):
        items = re.findall(r'<li>(.*?)</li>', m.group(0), re.S)
        if len(items) >= 3 and all(re.fullmatch(r'\s*<a href="#[^"]+">[^<]*</a>\s*', i) for i in items):
            return ''
        return m.group(0)
    return re.sub(r'<(ul|ol)>.*?</\1>', repl, fragment, flags=re.S)


def split_sections(text, ch):
    """원고를 (절 제목, HTML) 목록으로 나눈다. 첫 '##' 앞 내용은 00 개요 절이 된다."""
    is_md = ch['source'].endswith('.md')
    text = text.replace('\r\n', '\n')
    if is_md:
        text = re.sub(r'^#\s+.+\n', '', text, count=1, flags=re.M)  # 장 제목(# …)은 상단 배너로 옮김
        # 코드 블록 안의 '## '은 절로 자르지 않도록 보호
        guard = {}
        def keep(m):
            key = f'\x00CODE{len(guard)}\x00'
            guard[key] = m.group(0)
            return key
        text = re.sub(r'^(```|~~~).*?^\1\s*$', keep, text, flags=re.S | re.M)
    parts = re.split(r'^##\s+(.+?)\s*#*\s*$', text, flags=re.M)
    intro, rest = parts[0].strip(), parts[1:]
    raw = [(rest[i].strip(), rest[i + 1].strip()) for i in range(0, len(rest), 2)]
    raw = [(t, b) for t, b in raw if strip_number(t) not in SKIP and t not in SKIP]
    # 'base': 원고를 옮겨 왔을 때 원래 HTML이 있던 폴더 — 본문의 상대 경로는 그 위치 기준
    src_dir = os.path.join(ROOT, ch['base']) if ch.get('base') else os.path.dirname(os.path.join(ROOT, ch['source']))
    def render(body):
        if is_md:
            for k, v in guard.items():
                body = body.replace(k, v)
            body = code_boxes(MARKDOWN(body))
        return fix_links(body, src_dir)
    secs = []
    intro_title = ch.get('intro_title', '개요')
    intro_html = drop_local_toc(render(intro)) if intro else ''
    if not intro_html.strip():
        intro_html = f'<p>{escape(ch["subtitle"])}</p>'
    secs.append((intro_title, intro_html))
    for t, b in raw:
        title = strip_number(re.sub(r'<[^>]+>', '', MARKDOWN(t)).strip() if is_md else t)
        body = render(b)
        if nav_only(body):
            continue
        secs.append((html.unescape(title), body, slug_variants(t)))
    # 원고 안 옛 제목 앵커(#01--제목 등) → 새 절 번호(#sNN)
    alias = {}
    for i, s in enumerate(secs):
        for v in (s[2] if len(s) > 2 else ()):
            alias.setdefault(v, f's{i:02d}')
    def re_anchor(m):
        key = urllib.parse.unquote(m.group(1)).lower()
        return f'href="#{alias[key]}"' if key in alias else m.group(0)
    return [(s[0], re.sub(r'href="#([^"]+)"', re_anchor, s[1])) for s in secs]


def page(title, kicker, desc, chips, body, chapter_id=''):
    manifest = {'title': CONFIG['site_title'], 'short_title': CONFIG['course_code'],
                'home_label': '전체 학습 안내서',
                'chapters': [{'id': c['id'], 'number': c['number'], 'title': c['nav_title'], 'file': c['file']}
                             for c in CHAPTERS]}
    course_data = json.dumps(manifest, ensure_ascii=False).replace('<', '\\u003c')
    chips_html = ''.join(f'<span class="chip">{escape(c)}</span>' for c in chips)
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{escape(desc)}"><title>{escape(title)} · {escape(CONFIG['site_title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body data-chapter="{escape(chapter_id)}">
<a class="skip-link" href="#main-content">본문 바로가기</a>
<button type="button" class="mobile-menu" id="mobile-menu" aria-controls="course-nav" aria-expanded="false">☰ 장·목차</button><div class="shade" id="shade"></div>
<header class="hero"><div class="hero-inner"><p class="kicker">{escape(kicker)}</p><h1>{escape(title)}</h1><p>{escape(desc)}</p><div class="chips">{chips_html}</div></div></header>
<div class="wrap"><nav class="toc" id="course-nav" aria-label="장과 절 탐색"></nav><main class="content" id="main-content">
{body}
<div class="chapter-links" id="chapter-links"></div></main></div>
<footer class="footer">{escape(CONFIG['site_title'])} · {escape(CONFIG['version'])} · <a href="{RULE_URL}">rule_md 교육용 HTML 표준</a> 적용 · {escape(CONFIG.get('footer_note', '실습 결과는 실제 환경에서 검증'))}</footer>
<script>window.COURSE={course_data};</script><script>{SCRIPT}</script></body></html>
'''


def section(n, title, inner):
    return (f'<section class="section" id="s{n:02d}"><h2><span class="num">{n:02d}</span>{escape(title)}</h2>'
            f'<div class="sec-body">\n{inner}\n</div></section>')


def legacy_redirect(ch, legacy):
    """이전 주소로 들어온 방문자를 새 장으로 보낸다."""
    target = os.path.relpath(OUT / ch['file'], (OUT / legacy).parent).replace(os.sep, '/')
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="robots" content="noindex">
<title>이동 안내 · 제{label(ch['number'])}장 {escape(ch['title'])}</title>
<link rel="canonical" href="{escape(target)}">
<meta http-equiv="refresh" content="0; url={escape(target)}">
<script>location.replace({json.dumps(target)});</script>
</head><body><p>이 장은 새 주소로 옮겨졌습니다: <a href="{escape(target)}">제{label(ch['number'])}장 · {escape(ch['title'])}</a></p></body></html>
'''


def build_index(toc):
    parts = CONFIG.get('parts') or [{'title': '전체 과정', 'subtitle': CONFIG['site_subtitle'],
                                     'chapters': [c['number'] for c in CHAPTERS]}]
    flow, grids, details = [], [], []
    for p in parts:
        chs = [BY_NUM[n] for n in p['chapters']]
        flow.append(f'<div class="step"><strong>{escape(p["title"])}</strong> — {escape(p["subtitle"])}<br>'
                    + ' → '.join(f'<a href="{escape(c["file"])}">{label(c["number"])} {escape(c["nav_title"])}</a>' for c in chs)
                    + '</div>')
        cards = ''.join(f'<a class="chapter-card" href="{escape(c["file"])}"><span>CHAPTER {label(c["number"])}</span>'
                        f'<b>{escape(c["title"])}</b><span>{escape(c["subtitle"])}</span></a>' for c in chs)
        head = f'<h3 class="part">{escape(p["title"])}<small>{escape(p["subtitle"])}</small></h3>' if len(parts) > 1 else ''
        grids.append(f'{head}<div class="chapter-grid">{cards}</div>')
    for c in CHAPTERS:
        items = ''.join(f'<li><a href="{escape(c["file"])}#s{i:02d}"><em>{i:02d}</em><span>{escape(t)}</span></a></li>'
                        for i, t in enumerate(toc[c['id']]))
        details.append(f'<details class="page-list"><summary>제{label(c["number"])}장 · {escape(c["title"])} — {len(toc[c["id"]])}개 절</summary><ol>{items}</ol></details>')
    total = sum(len(v) for v in toc.values())
    cards_cfg = CONFIG.get('overview_cards', [])
    cards = ''.join(f'<div class="card"><b>{escape(k["label"])}</b><span class="val">{escape(k["value"])}</span><span>{escape(k["note"])}</span></div>'
                    for k in [{'label': '구성', 'value': f'{len(CHAPTERS)}장', 'note': f'{len(parts)}개 부(部)'},
                              {'label': '분량', 'value': f'{total}절', 'note': '장마다 개요 절 포함'}] + cards_cfg)
    notes = ''.join(f'<div class="note"><span class="lbl">안내</span>{escape(n)}</div>' for n in CONFIG.get('notes', []))
    extra = ''.join(f'<p>{escape(x)}</p>' for x in CONFIG.get('principles', []))
    links = ''.join(f'<li><a href="{escape(l["href"])}">{escape(l["text"])}</a></li>' for l in CONFIG.get('extra_links', []))
    extra_links = f'<h3>함께 보는 자료</h3><ul>{links}</ul>' if links else ''
    content = f'''<section class="section" id="s00"><h2><span class="num">00</span>과정 개요</h2><div class="sec-body">
<p>{escape(CONFIG['site_description'])}</p><div class="cards">{cards}</div>{notes}</div></section>
<section class="section" id="s01"><h2><span class="num">01</span>전체 학습 테크트리</h2><div class="sec-body"><p>앞 장의 내용이 다음 장의 바탕이 되므로 순서대로 진행합니다.</p><div class="flow">{'<div class="down">↓</div>'.join(flow)}</div></div></section>
<section class="section" id="s02"><h2><span class="num">02</span>장별 목차</h2><div class="sec-body">{''.join(grids)}</div></section>
<section class="section" id="s03"><h2><span class="num">03</span>장별 세부 목차</h2><div class="sec-body"><p>절 제목을 누르면 해당 장의 그 절로 바로 이동합니다.</p>{''.join(details)}</div></section>
<section class="section" id="s04"><h2><span class="num">04</span>학습·검증 원칙</h2><div class="sec-body"><p>각 장은 개요 → 개념 → 실습 코드 → 예상 출력 → 검증 순서로 읽습니다. 실제 실행 결과와 교육용 예시는 구분하며, 실행하지 않은 항목은 <strong>미검증</strong>으로 표시합니다.</p>{extra}{extra_links}</div></section>'''
    chips = CONFIG.get('index_chips', [f'{len(CHAPTERS)}개 장', f'{total}개 절'])
    return page(CONFIG['site_title'] + (' — ' + CONFIG['site_subtitle'] if CONFIG.get('site_subtitle') else ''),
                CONFIG['course_code'], CONFIG['site_description'], chips, content)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    expected, toc = {OUT / 'index.html'}, {}
    for ch in CHAPTERS:
        src = (ROOT / ch['source']).resolve()
        if ROOT not in src.parents or not src.is_file():
            raise ValueError('잘못된 본문 파일 경로: ' + ch['source'])
        secs = split_sections(src.read_text(encoding='utf-8'), ch)
        toc[ch['id']] = [t for t, _ in secs]
        body = '\n'.join(section(i, t, inner) for i, (t, inner) in enumerate(secs))
        out = page(f"제{label(ch['number'])}장 · {ch['title']}",
                   f"{CONFIG['course_code']} · CHAPTER {label(ch['number'])}",
                   ch['subtitle'], ch.get('tags', []), body, ch['id'])
        (OUT / ch['file']).write_text(out, encoding='utf-8')
        expected.add(OUT / ch['file'])
        if ch.get('legacy_file'):
            p = ROOT / ch['legacy_file']
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(legacy_redirect(ch, os.path.relpath(p, OUT)), encoding='utf-8')
    (OUT / 'index.html').write_text(build_index(toc), encoding='utf-8')
    # 최신 버전만 남기기: 출력 폴더의 생성본 중 현재 목록에 없는 이전 _vNN 파일만 제거
    pattern = re.compile(r'^(' + '|'.join(re.escape(c['slug']) for c in CHAPTERS) + r')_v\d+\.html$')
    for p in OUT.glob('*_v*.html'):
        if pattern.match(p.name) and p not in expected:
            p.unlink()
    print('생성 완료:', len(expected), '개 HTML')
    for p in sorted(expected):
        print(' ', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    build()
