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
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
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
    """RSS 엔트리에서 대표 이미지 URL 추출 (원본 기사 이미지 우선)"""
    # 1. media_content (가장 신뢰도 높음)
    media = entry.get("media_content", [])
    if media:
        for m in media:
            url = m.get("url", "")
            if url and any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                return url
        if media[0].get("url"):
            return media[0]["url"]

    # 2. media_thumbnail
    thumbs = entry.get("media_thumbnail", [])
    if thumbs and thumbs[0].get("url"):
        return thumbs[0]["url"]

    # 3. enclosure (일부 RSS에서 사용)
    enclosures = entry.get("enclosures", [])
    for enc in enclosures:
        enc_type = enc.get("type", "")
        if enc_type.startswith("image/") and enc.get("href"):
            return enc["href"]

    links = entry.get("links", [])
    for link in links:
        if link.get("type", "").startswith("image/") and link.get("href"):
            return link["href"]

    # 4. content / summary HTML 에서 <img> 추출
    html_parts = []
    content_list = entry.get("content", [])
    if content_list:
        for c in content_list:
            html_parts.append(c.get("value", ""))
    html_parts.append(entry.get("summary", ""))
    full_html = " ".join(html_parts)

    img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', full_html)
    for img_url in img_matches:
        if any(skip in img_url.lower() for skip in ["tracking", "pixel", "1x1", "spacer", "blank", "avatar", "icon", "logo", "badge"]):
            continue
        return img_url

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
      "word_map": {
        "english_word": "한글뜻",
        "another_word": "다른뜻"
      },
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
- Summarize the article into EXACTLY 2 paragraphs. Each paragraph must contain 3-4 sentences.
- The content array must contain 6 to 8 sentence pair objects in total. No more, no less.
- Each sentence pair is one sentence — do NOT merge multiple sentences into one object.
- word_map: For each sentence pair, extract 5-8 key vocabulary words that learners should know. Keys are lowercase English words from the sentence, values are the corresponding Korean translation AS USED IN THIS SPECIFIC SENTENCE CONTEXT. Example: {"assembled": "구축한", "portfolio": "포트폴리오", "diverse": "다양한"}. The mapping must reflect how the word is actually translated in the ko sentence, not a generic dictionary definition.
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
    """RSS 이미지가 있으면 사용, 없으면 null (부적합 랜덤 이미지 제거)"""
    if entry.get("image_url"):
        return entry["image_url"]
    return None


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

    targets = new_entries[:MAX_ARTICLES * 2]
    if not targets:
        print("[SKIP] 처리할 새로운 기사가 없습니다.")
        return

    print(f"[PLAN] 최대 {len(targets)}개 중 {MAX_ARTICLES}개 성공 목표")

    # LLM 처리: 성공한 기사만 모아서 정확히 3개까지만
    ready = []

    for i, entry in enumerate(targets):
        if len(ready) >= MAX_ARTICLES:
            break

        print(f"[{i+1}/{len(targets)}] 처리 중: {entry['title']}")

        try:
            data = generate_article_data(entry)
            print(f"  [LLM] 변환 완료 — 문장 {len(data.get('content',[]))}개")
            ready.append((entry, data))
        except Exception as e:
            print(f"  [ERROR/LLM] {e}")
            if i < len(targets) - 1:
                print(f"  [WAIT] {API_COOLDOWN}초 대기 후 다음 기사로...")
                time.sleep(API_COOLDOWN)
            continue

        if len(ready) < MAX_ARTICLES and i < len(targets) - 1:
            print(f"  [WAIT] {API_COOLDOWN}초 대기...")
            time.sleep(API_COOLDOWN)

    if not ready:
        print("[SKIP] LLM 처리에 성공한 기사가 없습니다.")
        return

    print(f"\n[READY] 성공 {len(ready)}개 — DB 초기화 후 Insert 시작")

    # 기존 데이터 전체 삭제
    try:
        sb.table("articles").delete().neq("id", 0).execute()
        print("[DB] 기존 articles 전체 삭제 완료")
    except Exception as e:
        print(f"[ERROR/DB] 기존 데이터 삭제 실패: {e}")
        return

    # 성공한 기사만 Insert
    success = 0
    for entry, data in ready:
        try:
            save_to_supabase(entry, data)
            print(f"  [DB] 저장 완료: {data['title_ko']}")
            success += 1
        except Exception as e:
            print(f"  [ERROR/DB] {e}")

    print(f"\n[IEL Pipeline] 완료 — {success}개 저장")


if __name__ == "__main__":
    run()
