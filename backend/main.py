"""IEL Pipeline — RSS 수집 → LLM 변환 → Supabase 저장"""

import os
import re
import json
import time
import feedparser
import requests
from datetime import datetime, timezone
from openai import OpenAI
from supabase import create_client

# ── Config ──
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

RSS_FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",
]

MAX_ARTICLES = 3
API_COOLDOWN = 20
IMAGE_SKIP = ["tracking", "pixel", "1x1", "spacer", "blank", "avatar", "icon", "logo", "badge"]
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IELBot/1.0)"}

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
llm = OpenAI(api_key=OPENAI_API_KEY)

# 토큰 최적화를 위해 최소한의 구조 + 규칙만 전달
SYSTEM_PROMPT = """You are an EN-KO bilingual IT education content creator.
Return a JSON object for the given article:

{"title_ko":"한글제목","summary_ko":"200자이내 한글요약","difficulty":5,"image_keyword":"topic","content":[{"en":"English.","ko":"한글.","word_map":{"word":"뜻"},"feedback_en_to_ko":{"ideal":"자연스러운 번역","comments":["팁1","팁2"]},"feedback_ko_to_en":{"ideal":"Natural translation.","comments":["tip1","tip2"]}}]}

Rules:
1. content: exactly 6-8 sentence pair objects (one sentence each, never merge)
2. word_map: 5-8 lowercase EN keys → KO values per sentence, context-specific (not dictionary definitions)
3. difficulty: 1-10 scale
4. feedback.ideal: a MORE NATURAL alternative — not the only correct answer. Many translations are valid.
5. feedback.comments: 2 items. First comment = what makes a good translation of this sentence (praise the key concept). Second comment = a specific nuance tip (word choice, register, or grammar point). Never say the user is wrong — offer improvement suggestions.
6. Valid JSON only, no markdown fences"""


def extract_image(entry):
    """RSS 엔트리에서 이미지 URL 추출 (media → thumbnail → enclosure → HTML)"""
    for m in entry.get("media_content", []):
        url = m.get("url", "")
        if url:
            return url

    thumbs = entry.get("media_thumbnail", [])
    if thumbs and thumbs[0].get("url"):
        return thumbs[0]["url"]

    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("image/") and enc.get("href"):
            return enc["href"]

    for link in entry.get("links", []):
        if link.get("type", "").startswith("image/") and link.get("href"):
            return link["href"]

    # HTML <img> 추출 (트래킹 픽셀 필터링)
    html_parts = [c.get("value", "") for c in entry.get("content", [])]
    html_parts.append(entry.get("summary", ""))
    for img_url in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', " ".join(html_parts)):
        if not any(skip in img_url.lower() for skip in IMAGE_SKIP):
            return img_url

    return None


def fetch_og_image(url):
    """기사 페이지에서 og:image 메타태그 추출"""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=10, headers=HTTP_HEADERS)
        resp.raise_for_status()
        match = re.search(
            r'<meta[^>]+(?:property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']'
            r'|content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\'])',
            resp.text,
        )
        if match:
            img = match.group(1) or match.group(2)
            if img and img.startswith("http"):
                print(f"  [OG] {img[:80]}")
                return img
    except Exception as e:
        print(f"  [OG] 실패: {e}")
    return None


def resolve_image(entry):
    """RSS 이미지 → og:image → None"""
    return entry.get("image_url") or fetch_og_image(entry.get("link", ""))


def fetch_rss_entries():
    """RSS 피드에서 최신 기사 수집"""
    entries = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.get("title", feed_url)
            for e in feed.entries[:5]:
                entries.append({
                    "title": e.get("title", ""),
                    "link": e.get("link", ""),
                    "summary": e.get("summary", ""),
                    "source": source,
                    "image_url": extract_image(e),
                })
        except Exception as e:
            print(f"[RSS] {feed_url} 실패: {e}")
    return entries


def filter_new(entries):
    """DB에 이미 있는 기사 제외"""
    try:
        existing = sb.table("articles").select("source_url").execute()
        urls = {r["source_url"] for r in (existing.data or [])}
    except Exception as e:
        print(f"[DB] 조회 실패: {e}")
        urls = set()
    return [e for e in entries if e["link"] not in urls]


def generate(entry):
    """LLM으로 학습 콘텐츠 생성"""
    resp = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Title: {entry['title']}\n"
                f"URL: {entry['link']}\n"
                f"Source: {entry['source']}\n"
                f"Summary: {entry['summary'][:500]}"
            )},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def save(entry, data):
    """Supabase에 기사 저장"""
    sb.table("articles").insert({
        "title": data["title_ko"],
        "summary": data["summary_ko"],
        "difficulty": data["difficulty"],
        "source_name": entry["source"],
        "source_url": entry["link"],
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "content": json.dumps(data["content"], ensure_ascii=False),
        "image_url": resolve_image(entry),
    }).execute()


def run():
    print(f"[IEL] 시작 — {datetime.now(timezone.utc).isoformat()}")

    entries = fetch_rss_entries()
    print(f"[RSS] {len(entries)}개 수집")

    targets = filter_new(entries)[:MAX_ARTICLES * 2]
    if not targets:
        print("[SKIP] 신규 기사 없음")
        return

    # LLM 변환 (최대 MAX_ARTICLES개 성공할 때까지)
    ready = []
    for i, entry in enumerate(targets):
        if len(ready) >= MAX_ARTICLES:
            break
        print(f"[{i+1}/{len(targets)}] {entry['title']}")
        try:
            data = generate(entry)
            print(f"  [LLM] {len(data.get('content', []))}문장")
            ready.append((entry, data))
        except Exception as e:
            print(f"  [ERR] {e}")

        if i < len(targets) - 1 and len(ready) < MAX_ARTICLES:
            time.sleep(API_COOLDOWN)

    if not ready:
        print("[SKIP] LLM 성공 0건")
        return

    # DB 교체: 전체 삭제 → 새 기사 삽입
    print(f"\n[DB] {len(ready)}개 저장 시작")
    try:
        sb.table("articles").delete().neq("id", 0).execute()
    except Exception as e:
        print(f"[ERR] 삭제 실패: {e}")
        return

    ok = 0
    for entry, data in ready:
        try:
            save(entry, data)
            print(f"  [OK] {data['title_ko']}")
            ok += 1
        except Exception as e:
            print(f"  [ERR] {e}")

    print(f"\n[IEL] 완료 — {ok}/{len(ready)}개 저장")


if __name__ == "__main__":
    run()
