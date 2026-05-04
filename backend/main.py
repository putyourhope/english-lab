"""
IEL 백엔드 파이프라인
- RSS에서 IT 기사 수집
- LLM으로 en/ko 문장 페어 + feedback 생성
- Supabase articles 테이블에 Insert
"""

import os
import re
import json
import time
import feedparser
from datetime import datetime, timezone
from openai import OpenAI
from supabase import create_client

# ── 환경변수 ──
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # service_role key (서버 전용)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
llm = OpenAI(api_key=OPENAI_API_KEY)

# ── RSS 피드 목록 ──
RSS_FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",
]

MAX_ARTICLES = 3
API_COOLDOWN = 20  # 초


def fetch_rss_entries():
    """RSS 피드에서 최신 기사 목록 수집"""
    entries = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                entries.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "source": feed.feed.get("title", url),
                    "published": entry.get("published", ""),
                    "image_url": extract_image(entry),
                })
        except Exception as e:
            print(f"[RSS] {url} 실패: {e}")
    return entries


def extract_image(entry):
    """RSS 엔트리에서 대표 이미지 URL 추출"""
    # media_content
    media = entry.get("media_content", [])
    if media:
        for m in media:
            url = m.get("url", "")
            if url and any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                return url
        if media[0].get("url"):
            return media[0]["url"]

    # media_thumbnail
    thumbs = entry.get("media_thumbnail", [])
    if thumbs and thumbs[0].get("url"):
        return thumbs[0]["url"]

    # og:image or <img> from summary/content HTML
    html = entry.get("summary", "") + entry.get("content", [{}])[0].get("value", "") if entry.get("content") else entry.get("summary", "")
    img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html)
    if img_match:
        return img_match.group(1)

    return None


def filter_new_entries(entries):
    """이미 DB에 있는 기사를 제외"""
    try:
        existing = sb.table("articles").select("source_url").execute()
        existing_urls = {r["source_url"] for r in (existing.data or [])}
    except Exception as e:
        print(f"[DB] 기존 기사 조회 실패: {e}")
        existing_urls = set()

    return [e for e in entries if e["link"] not in existing_urls]


SYSTEM_PROMPT = """You are an English-Korean bilingual education content creator for IT professionals.
Given an article title, link, and summary, produce a JSON object with this exact structure:

{
  "title_ko": "한글 제목",
  "summary_ko": "200자 이내 한글 요약",
  "difficulty": 5,
  "image_keyword": "one or two English words for article topic",
  "content": [
    {
      "en": "Original English sentence.",
      "ko": "한글 번역.",
      "feedback_en_to_ko": {
        "ideal": "더 자연스러운 한글 번역 모범 답안",
        "comments": ["번역 팁 1", "번역 팁 2"]
      },
      "feedback_ko_to_en": {
        "ideal": "More natural English back-translation.",
        "comments": ["영작 팁 1", "영작 팁 2"]
      }
    }
  ]
}

CRITICAL RULES:
- Extract EXACTLY 2 key sentences from the article. No more, no less. The content array must have exactly 2 objects.
- difficulty is 1-10 based on vocabulary/grammar complexity.
- Each feedback has exactly 2 comments.
- ideal translations should differ meaningfully from the literal ko/en to teach nuance.
- image_keyword: 1-2 simple English words describing the article's core topic (e.g. "artificial intelligence", "remote work", "cybersecurity"). Used as a fallback image search term.
- All output must be valid JSON. No markdown fences."""


def generate_article_data(entry):
    """LLM으로 기사를 학습용 데이터로 변환"""
    user_msg = f"""Title: {entry['title']}
URL: {entry['link']}
Source: {entry['source']}
Summary: {entry['summary'][:500]}

Please create the IEL learning content JSON for this article."""

    resp = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


def resolve_image_url(entry, data):
    """RSS 이미지 → LLM 키워드 fallback"""
    if entry.get("image_url"):
        return entry["image_url"]
    keyword = data.get("image_keyword", "technology")
    safe_kw = keyword.replace(" ", "+")
    return f"https://loremflickr.com/800/400/{safe_kw}"


def save_to_supabase(entry, data):
    """Supabase articles 테이블에 저장"""
    row = {
        "title": data["title_ko"],
        "summary": data["summary_ko"],
        "difficulty": data["difficulty"],
        "source_name": entry["source"],
        "source_url": entry["link"],
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "content": json.dumps(data["content"], ensure_ascii=False),
        "image_url": resolve_image_url(entry, data),
    }

    result = sb.table("articles").insert(row).execute()
    return result


def run():
    print(f"[IEL Pipeline] 시작 — {datetime.now(timezone.utc).isoformat()}")

    entries = fetch_rss_entries()
    print(f"[RSS] 총 {len(entries)}개 기사 수집")

    new_entries = filter_new_entries(entries)
    print(f"[FILTER] 신규 기사 {len(new_entries)}개")

    targets = new_entries[:MAX_ARTICLES]
    if not targets:
        print("[SKIP] 처리할 새로운 기사가 없습니다.")
        return

    print(f"[PLAN] {len(targets)}개 기사 처리 예정")

    try:
        sb.table("articles").delete().gte("id", 0).execute()
        print("[DB] 기존 articles 전체 삭제 완료 (덮어쓰기 준비)\n")
    except Exception as e:
        print(f"[ERROR/DB] 기존 데이터 삭제 실패: {e}\n")

    success = 0
    fail = 0

    for i, entry in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] 처리 중: {entry['title']}")

        try:
            data = generate_article_data(entry)
            print(f"  [LLM] 변환 완료 — 문장 {len(data.get('content',[]))}개")
        except Exception as e:
            print(f"  [ERROR/LLM] {e}")
            fail += 1
            if i < len(targets) - 1:
                print(f"  [WAIT] {API_COOLDOWN}초 대기 후 다음 기사로...")
                time.sleep(API_COOLDOWN)
            continue

        try:
            save_to_supabase(entry, data)
            print(f"  [DB] 저장 완료: {data['title_ko']}")
            success += 1
        except Exception as e:
            print(f"  [ERROR/DB] {e}")
            fail += 1

        if i < len(targets) - 1:
            print(f"  [WAIT] {API_COOLDOWN}초 대기...")
            time.sleep(API_COOLDOWN)

    print(f"\n[IEL Pipeline] 완료 — 성공 {success}개 / 실패 {fail}개")


if __name__ == "__main__":
    run()
