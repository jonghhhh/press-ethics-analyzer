"""
뉴스 기사 스크래핑 도구 (최적화 버전)

사용법:
    article = extract_article(url)

반환 형식 (JSON):
    {
        'title': '기사 제목',
        'text': '기사 본문 텍스트',
        'image_url': '대표 이미지 URL'
    }

의존성: pip3 install trafilatura newspaper3k playwright beautifulsoup4 requests fake-useragent extruct
Playwright 초기 설치: playwright install chromium

성능 최적 순서:
1. Trafilatura (가장 빠르고 정확, 정적 콘텐츠)
2. Newspaper3k (빠르고 한국어 지원 우수)
3. Playwright + Trafilatura (JavaScript 렌더링 필요시)
4. Playwright + Newspaper3k (대체 방법)
"""

import json
import time
from typing import Optional, Dict
from urllib.parse import urljoin

import requests
import trafilatura
from bs4 import BeautifulSoup
from newspaper import Article

try:
    from fake_useragent import UserAgent
    ua = UserAgent()
    USER_AGENT = ua.random
except ImportError:
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright 미설치 - JavaScript 렌더링 기능 비활성화")

try:
    import extruct
    EXTRUCT_AVAILABLE = True
except ImportError:
    EXTRUCT_AVAILABLE = False

# HTTP 헤더 설정
HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def fetch_with_headers(url: str) -> str:
    """HTTP 헤더를 포함한 URL 요청"""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text

def extract_images_from_html(html: str, base_url: str = "") -> Optional[str]:
    """HTML에서 이미지 추출 (여러 방법 시도)"""
    soup = BeautifulSoup(html, 'html.parser')

    # 1. og:image 메타태그
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        return og_image.get('content')

    # 2. twitter:image
    tw_image = soup.find('meta', attrs={'name': 'twitter:image'})
    if tw_image and tw_image.get('content'):
        return tw_image.get('content')

    # 3. extruct로 JSON-LD 파싱
    if EXTRUCT_AVAILABLE:
        try:
            metadata = extruct.extract(html, base_url=base_url)
            # Schema.org ImageObject 찾기
            for item in metadata.get('json-ld', []):
                if isinstance(item, dict):
                    if item.get('image'):
                        img = item['image']
                        if isinstance(img, str):
                            return img
                        elif isinstance(img, dict) and img.get('url'):
                            return img['url']
                        elif isinstance(img, list) and len(img) > 0:
                            return img[0] if isinstance(img[0], str) else img[0].get('url')
        except:
            pass

    # 4. article 내부의 첫 번째 이미지
    article_imgs = soup.select('article img[src], .article img[src], #article img[src]')
    if article_imgs:
        src = article_imgs[0].get('src')
        return urljoin(base_url, src) if src else None

    # 5. 일반 img 태그
    imgs = soup.find_all('img', src=True)
    for img in imgs:
        src = img.get('src')
        # 로고, 아이콘 제외
        if src and not any(x in src.lower() for x in ['logo', 'icon', 'avatar', 'profile', 'ad', 'banner']):
            # 최소 크기 확인 (width/height 속성)
            width = img.get('width', '0')
            height = img.get('height', '0')
            try:
                if int(width) >= 200 or int(height) >= 200:
                    return urljoin(base_url, src)
            except:
                return urljoin(base_url, src)

    return None

def extract_trafilatura(url: str) -> Optional[Dict[str, str]]:
    """Trafilatura 기사 추출"""
    try:
        html = fetch_with_headers(url)
        result = trafilatura.extract(html, output_format='json', url=url,
                                    include_images=True, include_links=True)
        if result:
            data = json.loads(result)
            image_url = data.get('image') or extract_images_from_html(html, url)

            return {
                'title': data.get('title'),
                'text': data.get('text'),
                'image_url': image_url
            }
    except Exception as e:
        print(f"Trafilatura 실패: {e}")
    return None

def extract_newspaper(url: str) -> Optional[Dict[str, str]]:
    """Newspaper3k 기사 추출"""
    try:
        html = fetch_with_headers(url)
        article = Article(url)
        article.config.browser_user_agent = HEADERS['User-Agent']
        article.set_html(html)
        article.parse()

        image_url = article.top_image or extract_images_from_html(html, url)

        return {
            'title': article.title,
            'text': article.text,
            'image_url': image_url
        }
    except Exception as e:
        print(f"Newspaper3k 실패: {e}")
    return None

def get_rendered_html_playwright(url: str, wait: int = 2) -> Optional[str]:
    """Playwright로 렌더링된 HTML 가져오기"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=HEADERS['User-Agent'],
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            time.sleep(wait)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"Playwright 오류: {e}")
        return None

def extract_playwright_trafilatura(url: str) -> Optional[Dict[str, str]]:
    """Playwright + Trafilatura 조합"""
    try:
        html = get_rendered_html_playwright(url)
        if html:
            result = trafilatura.extract(html, output_format='json', url=url,
                                        include_images=True, include_links=True)
            if result:
                data = json.loads(result)
                image_url = data.get('image') or extract_images_from_html(html, url)

                return {
                    'title': data.get('title'),
                    'text': data.get('text'),
                    'image_url': image_url
                }
    except Exception as e:
        print(f"Playwright+Trafilatura 실패: {e}")
    return None

def extract_playwright_newspaper(url: str) -> Optional[Dict[str, str]]:
    """Playwright + Newspaper3k 조합"""
    try:
        html = get_rendered_html_playwright(url)
        if html:
            article = Article(url='')
            article.set_html(html)
            article.parse()

            image_url = article.top_image or extract_images_from_html(html, url)

            return {
                'title': article.title,
                'text': article.text,
                'image_url': image_url
            }
    except Exception as e:
        print(f"Playwright+Newspaper3k 실패: {e}")
    return None


def extract_article(url: str) -> Optional[Dict[str, str]]:
    """기사 추출 - 최적 순서로 시도"""
    print(f"🔍 추출 시작: {url}")

    result = {'title': None, 'text': None, 'image_url': None}

    # 최적 순서: 빠르고 정확한 것부터 시도
    # 1. Trafilatura - 가장 빠르고 정확 (정적 콘텐츠)
    # 2. Newspaper3k - 빠르고 한국어 지원 우수
    # 3. Playwright + Trafilatura - JavaScript 렌더링이 필요한 경우
    # 4. Playwright + Newspaper3k - 대체 방법
    extractors = [
        ("Trafilatura", extract_trafilatura),
        ("Newspaper3k", extract_newspaper),
    ]

    # Playwright 추가 (JavaScript 렌더링 필요시)
    if PLAYWRIGHT_AVAILABLE:
        extractors.extend([
            ("Playwright+Trafilatura", extract_playwright_trafilatura),
            ("Playwright+Newspaper3k", extract_playwright_newspaper),
        ])

    for i, (name, extractor) in enumerate(extractors, 1):
        print(f"   {i}️⃣ {name} 시도...")
        try:
            data = extractor(url)
            if data:
                # 결과 업데이트
                updated = []
                for key in result:
                    if not result[key] and data.get(key):
                        result[key] = data[key]
                        updated.append(key)

                if updated:
                    print(f"      → 추출 성공: {', '.join(updated)}")

                # 제목, 본문, 이미지 모두 있으면 성공
                if result['title'] and result['text'] and result['image_url']:
                    print(f"   ✅ {name} 완료! (제목 O, 본문 O, 이미지 O)")
                    return result

                # 상태 출력
                status = f"제목: {'O' if result['title'] else 'X'}, 본문: {'O' if result['text'] else 'X'}, 이미지: {'O' if result['image_url'] else 'X'}"
                if result['title'] and result['text']:
                    print(f"   ⚠️ 이미지 없음 - 다음 단계 계속 ({status})")
                else:
                    print(f"   ⚠️ 부분 성공 ({status})")
            else:
                print(f"   ❌ {name} 실패")
        except requests.HTTPError as e:
            if e.response.status_code in (403, 429):
                print(f"   ❌ {name} 차단됨 (HTTP {e.response.status_code})")
                raise
            print(f"   ❌ {name} 오류: {e}")
        except Exception as e:
            print(f"   ❌ {name} 오류: {e}")

    if result['title'] or result['text']:
        print(f"   ✅ 최종 결과 - 제목: {'O' if result['title'] else 'X'}, 본문: {'O' if result['text'] else 'X'}, 이미지: {'O' if result['image_url'] else 'X'}")
        return result

    print("   ❌ 모든 방법 실패")
    return None

if __name__ == "__main__":
    test_urls = [
        "https://www.chosun.com/national/education/2025/07/19/4OMZBICJSNDGXA567IKPRBUFKA/",
        "https://news.nate.com/view/20250521n37437",
        "https://www.hani.co.kr/arti/society/society_general/1204840.html"
    ]

    print(f"사용 가능한 도구:")
    print(f"  - Playwright: {'✅' if PLAYWRIGHT_AVAILABLE else '❌'}")
    print(f"  - Extruct: {'✅' if EXTRUCT_AVAILABLE else '❌'}")
    print(f"  - Fake UserAgent: {'✅' if 'ua' in dir() else '❌'}\n")

    for url in test_urls:
        print(f"\n{'='*60}")
        try:
            article = extract_article(url)
            if article:
                print(f"\n📄 제목: {article.get('title', 'N/A')[:100]}...")
                print(f"📝 본문: {len(article.get('text', ''))}자")
                print(f"🖼️ 이미지: {article.get('image_url', 'N/A')[:80]}..." if article.get('image_url') else "🖼️ 이미지: 없음")
            else:
                print("추출 실패")
        except Exception as e:
            print(f"전체 실패: {e}")
        print("="*60)
