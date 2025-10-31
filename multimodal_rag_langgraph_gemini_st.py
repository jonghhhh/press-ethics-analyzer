# -*- coding: utf-8 -*-
"""
Streamlit 기반 뉴스 심의문 분석 시스템 (Gemini 2.0 Flash 버전)
"""
import streamlit as st
import os
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import EmbeddingFunction
from news_text_scraper import extract_article
import base64
import requests
from PIL import Image
from io import BytesIO
import time

# ========== 페이지 설정 ==========
st.set_page_config(
    page_title="뉴스 심의문 분석 시스템",
    page_icon="📰",
    layout="wide"
)

# ========== 사이드바 설정 ==========
st.sidebar.title("⚙️ 설정")
st.sidebar.markdown("---")

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Google AI Studio에서 발급받은 Gemini API 키를 입력하세요."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 사용 방법
1. Gemini API Key 입력
2. 분석할 기사 URL 입력
3. '분석 시작' 버튼 클릭
4. 단계별 진행 상황 확인
5. 최종 결과 확인

### 주의사항
- API 키는 세션 종료 시 삭제됩니다
- 분석에는 수 분이 소요될 수 있습니다
""")

# ========== 메인 화면 ==========
st.title("📰 뉴스 심의문 분석 시스템")
st.markdown("**Gemini 2.0 Flash 기반 - 한국신문윤리위원회 심의 기준 적용**")
st.markdown("---")

# ========== 설정 ==========
CHROMA_PATH = "./chroma/"
COLLECTION_NAME = "press_ethics_e5_072025"

# ========== State 정의 ==========
class AnalysisState(TypedDict):
    url: str
    article: dict
    image_desc: str
    similar_cases: str
    decision: str
    review_result: dict
    error: str
    violation_count: int

# ========== 임베딩 함수 (캐싱) ==========
@st.cache_resource
def load_embedding_model():
    """임베딩 모델 로드 (캐싱)"""
    class CustomEmbedding(EmbeddingFunction):
        def __init__(self):
            self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device="cpu")

        def __call__(self, input):
            return self.model.encode(input).tolist()

    return CustomEmbedding()

@st.cache_resource
def load_chroma_collection():
    """ChromaDB 컬렉션 로드 (캐싱)"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings())
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"❌ ChromaDB 로드 실패: {e}")
        return None

# ========== 규정 및 프롬프트 ==========
REGULATION = """당신은 한국신문윤리위원회 심의위원입니다.
#신문윤리실천요강:
제1조「언론의 자유, 책임, 독립」①정치권력으로부터의 자유 ②사회·경제 세력으로부터의 독립 ③사회적 책임 ④차별과 편견 금지 ⑤사회적 약자 보호
제2조「취재 준칙」①신분 사칭·위장 금지 ②자료 무단 이용 금지 ③재난 및 사고 취재 ④전화 및 디지털 기기 활용 취재 ⑤도청 및 비밀촬영 금지 ⑥부당한 금전 제공 금지
제3조「보도 준칙」①보도기사의 사실과 의견 구분 ②공정 보도 ③반론의 기회 ④미확인 보도 명시 원칙 ⑤보도자료 검증 ⑥선정 보도 금지 ⑦재난 보도의 신중 ⑧자살 보도의 주의 ⑨피의사실 보도 ⑩표준어 사용
제4조「사법 보도 준칙」①재판 부당 영향 금지 ②판결문 등의 사전보도 금지
제5조「취재원의 명시와 보호」①취재원 보호 ②취재원 명시와 익명 조건 ③제3자 비방과 익명보도 금지 ④취재원과의 비보도 약속
제6조「보도유예 시한」①보도유예 시한 연장 금지 ②보도유예 시한의 효력 상실
제7조「범죄보도와 인권존중」①피의자 및 피고인의 명예 존중 ②피의자·피고인·참고인 등 촬영 신중 ③범죄와 무관한 가족 보호 ④성범죄 등의 2차 피해 방지 ⑤미성년 피의자 신원 보호
제8조「저작물의 전재와 인용」①통신기사의 출처 명시 ②타 언론사 보도 등의 표절 금지 ③출판물 등의 표절 금지 ④사진, 영상 등의 저작권 보호
제9조「평론의 원칙」①사설의 정론성 ②평론의 자유
제10조「편집 지침」①제목의 원칙 ②편집 변경 금지 ③기고문 변경 금지 ④기사 정정 ⑤관련사진 게재 ⑥사진 및 영상 조작 금지 ⑦기사와 광고의 구분 ⑧이용자의 권리 보호 ⑨부당한 재전송 금지
제11조「명예와 신용존중」①명예·신용 훼손 금지 ②사자의 명예 존중
제12조「사생활 보호」①사생활 침해 금지 ②개인정보 무단 검색 등 금지 ③사생활 등의 촬영 및 보도 금지 ④공인의 사생활 보도
제13조「청소년과 어린이 보호」①청소년과 어린이 취재 보도 ②범죄 보도와 청소년, 어린이 보호 ③유해환경으로부터의 보호 ④유괴·납치 보도제한 협조
제14조「정보의 부당이용금지」①소유 주식 등에 관한 보도 제한 ②주식·부동산 등의 부당 거래 금지
제15조「언론인의 품위」①금품수수 및 향응, 청탁 금지 ②부당한 집단 영향력 행사 금지 ③광고·판매 등 영업행위 금지
제16조「공익의 정의」①국가 안전 등 ②공중 안녕 ③범죄의 폭로 ④공중의 오도 방지"""

def parse_regulation_dict():
    """REGULATION을 파싱하여 조항 딕셔너리 생성"""
    articles = {}
    lines = REGULATION.split('\n')
    for line in lines:
        if line.startswith('제'):
            match = re.match(r'제(\d+)조「([^」]+)」(.+)', line)
            if match:
                num = match.group(1)
                name = match.group(2)
                items_text = match.group(3)
                items = {}
                item_pattern = r'([①②③④⑤⑥⑦⑧⑨⑩])([^①②③④⑤⑥⑦⑧⑨⑩]+)'
                for item_match in re.finditer(item_pattern, items_text):
                    item_num = item_match.group(1)
                    item_content = item_match.group(2).strip()
                    items[item_num] = item_content
                articles[num] = {'name': name, 'items': items}
    return articles

REGULATION_DICT = parse_regulation_dict()

def correct_article_reference(text):
    """심의문의 조항 참조를 REGULATION_DICT에 맞게 자동 수정"""
    pattern = r'제(\d+)조「([^」]+)」([①②③④⑤⑥⑦⑧⑨⑩])(?:항|호)?(?:\([^)]*\))*'

    def replace_match(match):
        article_num = match.group(1)
        cited_name = match.group(2).strip()
        item_num = match.group(3)

        if article_num in REGULATION_DICT:
            correct_name = REGULATION_DICT[article_num]['name']
            items = REGULATION_DICT[article_num]['items']

            if item_num in items:
                item_content = items[item_num]
                return f'제{article_num}조「{correct_name}」{item_num}({item_content})'
            else:
                return f'제{article_num}조「{correct_name}」{item_num}'
        return match.group(0)

    return re.sub(pattern, replace_match, text)

INST_PROMPT = """#심의 지침:
1. **보수적 판단 원칙**: 신문윤리실천요강을 체계적으로 검토하되, 매우 보수적으로 판단
2. **명백하고 심각한 위반만 지적**: 의심스럽거나 경미하거나 불분명한 사안은 모두 "위반 없음"
3. 유사 사례를 참고하되, 해당 기사의 구체적 내용과 맥락을 중심으로 독립적으로 판단
4. **특별 주의사항**:
   - 특정 단체/기업의 활동을 지나치게 칭찬하고 홍보하는 내용 → 제1조②(사회·경제 세력으로부터의 독립) 또는 제10조⑦(기사와 광고의 구분) 적용 검토
   - 단순히 단체 활동을 소개하는 수준은 위반 아님. 명백한 홍보/광고 목적이어야 함

#작성 형식 (반드시 정확히 준수):

**[위반 없음 시] - 절대 엄수:**
- 오직 "위반 없음" 글자만 출력. 어떠한 추가 설명, 이유, 코멘트도 절대 금지

**[위반 시] - 정확히 준수:**
아래 4단계를 반드시 순서대로 따르되, "1단계", "2단계" 등의 소제목 없이 자연스러운 문장으로 연결:
1단계) 기사 요약 2~3문장
   - "위 기사는 ○○○에 대해 보도하면서..." 형식으로 시작
2단계) 문제점 지적 1~2문장
   - "그러나 이 보도는...", "하지만..." 등으로 문제점 명확히 지적
3단계) 규정 근거 1~2문장
   - 신문윤리실천요강을 바탕으로 문제의 위반 정당성을 제시
4단계) 결론 문장 (정확히 이 형식 준수)
   - "따라서 위 보도는 신문윤리실천요강 제○조「조항명」○항(세부내용)을 위반했다고 인정하여 주문과 같이 결정한다."
- 전체 6문장 이상
- 유사 사례의 자연스러운 문장체 참고
- "1)", "2)", "3)" 등의 번호나 소제목 절대 사용 금지"""

# ========== Gemini API 호출 함수 ==========
def call_gemini(api_key: str, prompt: str, image_data: str = None, temperature: float = 0.0) -> str:
    """Gemini API 호출"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    generation_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=8192,
    )

    if image_data:
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64decode(image_data)
        }
        response = model.generate_content(
            [prompt, image_part],
            generation_config=generation_config
        )
    else:
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

    return response.text

# ========== 분석 함수 ==========
def analyze_article_streamlit(url: str, api_key: str, progress_container, status_container):
    """Streamlit용 기사 분석 함수"""

    # 진행 상황 표시
    progress_bar = progress_container.progress(0)

    # 1. 기사 추출
    status_container.info("🔍 1단계: 기사 추출 중...")
    progress_bar.progress(10)

    try:
        article = extract_article(url)
        if not article or not article.get('text'):
            status_container.error("❌ 기사 추출 실패: 유효한 기사를 찾을 수 없습니다.")
            return None
        status_container.success(f"✅ 기사 추출 완료: {article.get('title', '')[:50]}...")
    except Exception as e:
        status_container.error(f"❌ 기사 추출 오류: {e}")
        return None

    progress_bar.progress(20)

    # 2. 이미지 처리
    status_container.info("🖼️ 2단계: 이미지 처리 중...")
    image_desc = None
    img_url = article.get('image_url')

    if img_url:
        try:
            resp = requests.get(img_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            img = Image.open(BytesIO(resp.content))
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            image_desc = call_gemini(api_key, "이 이미지를 한국어로 상세히 설명해주세요.", image_data=b64, temperature=0.3)
            status_container.success("✅ 이미지 설명 생성 완료")
        except Exception as e:
            status_container.warning(f"⚠️ 이미지 처리 실패: {e}")
    else:
        status_container.info("ℹ️ 이미지 없음")

    progress_bar.progress(40)

    # 3. 유사 사례 검색
    status_container.info("🔎 3단계: 유사 사례 검색 중...")
    similar_cases = ""
    violation_count = 0
    no_violation_count = 0

    try:
        ef = load_embedding_model()
        collection = load_chroma_collection()

        if collection:
            text = f"{article.get('title', '')} {article.get('text', '')[:2000]}"
            query_emb = ef([text])
            results = collection.query(query_embeddings=query_emb, n_results=5)
            cases = []
            for i in range(len(results["documents"][0])):
                reason = results['metadatas'][0][i]['reason']
                cases.append(f"{i+1}. {reason}")

                # 위반 개수 카운팅
                if '위반' in reason and '위반 없음' not in reason and '위반없음' not in reason:
                    violation_count += 1
                elif '위반 없음' in reason or '위반없음' in reason:
                    no_violation_count += 1

            similar_cases = "\n".join(cases)
            status_container.success(f"✅ 유사 사례 {len(cases)}개 검색 완료 (위반 {violation_count}/5, 위반없음 {no_violation_count}/5)")
        else:
            status_container.warning("⚠️ 유사 사례 검색 실패: ChromaDB 로드 오류")
    except Exception as e:
        status_container.warning(f"⚠️ 유사 사례 검색 실패: {e}")

    progress_bar.progress(60)

    # 4. 심의문 생성
    status_container.info("📝 4단계: 심의문 생성 중...")

    try:
        prompt = f"{REGULATION}\n\n{INST_PROMPT}\n\n#기사:\n{article.get('title', '')} {article.get('text', '')[:2000]}"
        if image_desc:
            prompt += f"\n\n#이미지:\n{image_desc}"
        if similar_cases:
            prompt += f"\n\n#유사사례:\n{similar_cases}"

            if no_violation_count >= 4:
                prompt += f"\n\n**중요**: 유사 사례 5개 중 {no_violation_count}개가 '위반 없음'입니다. 4개 이상이므로 이 기사도 '위반 없음'을 강력하게 고려하십시오."

        decision = call_gemini(api_key, prompt, temperature=0.0)
        status_container.success("✅ 심의문 생성 완료")
    except Exception as e:
        status_container.error(f"❌ 심의문 생성 실패: {e}")
        return None

    progress_bar.progress(80)

    # 5. 최종 검토
    status_container.info("🔍 5단계: 최종 검토 중...")

    if "위반 없음" in decision or "위반없음" in decision:
        final_decision = "위반 없음"
        status_container.success("✅ 검토 완료: 위반 없음")
    else:
        try:
            review_prompt = f"""당신은 신문윤리위원회 검토 담당자입니다. 생성된 심의문을 검토하고 수정하세요.

#분석 대상 기사:
제목: {article.get('title', '')}
본문: {article.get('text', '')[:2000]}

#생성된 심의문:
{decision}

#신문윤리실천요강:
{REGULATION}

#검토 임무 (반드시 준수):
1. **조항 정확성**: 인용된 조항이 신문윤리실천요강에 정확히 존재하는지 확인(조항 번호, 조항명 대조)하고 틀린 부분 수정
2. **기사 관련성**: 심의문이 실제 기사 내용과 일치하는지 확인(환각 내용 삭제)하고 필요시 수정
3. **형식 검증 및 수정**:
   - "1)", "2)", "3)" 등의 번호나 소제목이 있으면 모두 삭제하고 자연스러운 문장체로 수정
   - 반드시: 기사 요약(2~3문장) → 문제점(1~2문장) → 근거(1~2문장) → 결론("따라서 위 보도는...") 순서 준수
4. **검토 의견 완전 제거**: "심의문에서 언급된...", "확인되지 않습니다", "검토 결과..." 등의 검토 의견을 절대 포함하지 말 것
   - 검토자의 메타적 코멘트는 모두 삭제
   - 오직 심의문 본문만 출력

수정된 최종 심의문만 출력하시오 (검토 의견 절대 포함 금지):"""

            final_decision = call_gemini(api_key, review_prompt, temperature=0.0)
            final_decision = correct_article_reference(final_decision.strip())
            status_container.success("✅ 검토 완료: 조항 정확성 및 기사 관련성 검증 완료")
        except Exception as e:
            status_container.warning(f"⚠️ 검토 실패: {e}")
            final_decision = decision

    progress_bar.progress(100)
    status_container.success("🎉 분석 완료!")

    # 결과 반환
    return {
        'article': article,
        'image_desc': image_desc,
        'similar_cases': similar_cases,
        'violation_count': violation_count,
        'no_violation_count': no_violation_count,
        'final_decision': final_decision
    }

# ========== 메인 UI ==========
url_input = st.text_input(
    "📎 기사 URL 입력",
    placeholder="https://news.example.com/article/12345",
    help="분석할 뉴스 기사의 URL을 입력하세요"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_button = st.button("🚀 분석 시작", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🔄 초기화", use_container_width=True)

if clear_button:
    st.rerun()

st.markdown("---")

# ========== 분석 실행 ==========
if analyze_button:
    if not gemini_api_key:
        st.error("❌ Gemini API Key를 입력해주세요!")
    elif not url_input:
        st.error("❌ 기사 URL을 입력해주세요!")
    else:
        # 진행 상황 컨테이너
        progress_container = st.container()
        status_container = st.container()

        # 분석 실행
        result = analyze_article_streamlit(url_input, gemini_api_key, progress_container, status_container)

        if result:
            st.markdown("---")
            st.header("📊 분석 결과")

            # 결과 표시
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📰 기사 정보")
                article = result['article']

                st.write(f"**제목:** {article.get('title', 'N/A')}")
                st.write(f"**언론사:** {article.get('media', 'N/A')}")
                st.write(f"**날짜:** {article.get('date', 'N/A')}")
                st.write(f"**URL:** {article.get('url', url_input)}")

                if article.get('image_url'):
                    st.write(f"**이미지 URL:** {article.get('image_url', 'N/A')}")
                    try:
                        st.image(article['image_url'], caption="기사 이미지", use_container_width=True)
                    except:
                        st.warning("이미지를 불러올 수 없습니다.")

                with st.expander("📄 기사 본문", expanded=False):
                    st.write(article.get('text', 'N/A')[:1000] + "..." if len(article.get('text', '')) > 1000 else article.get('text', 'N/A'))

            with col2:
                st.subheader("📈 분석 정보")

                # 분석 결과
                if result['final_decision'].strip() == "위반 없음":
                    st.success("✅ **분석 결과:** 위반 없음")
                else:
                    st.error("⚠️ **분석 결과:** 위반")

                # 유사 사례 통계
                st.metric("위반 사례", f"{result['violation_count']}/5")
                st.metric("위반 없음 사례", f"{result['no_violation_count']}/5")

            # 심의문
            st.markdown("---")
            st.subheader("⚖️ 최종 심의문")
            st.info(result['final_decision'])

            # 유사 사례
            if result['similar_cases']:
                with st.expander("📚 유사 사례 (5개)", expanded=False):
                    st.text(result['similar_cases'])

            # 이미지 설명
            if result['image_desc']:
                with st.expander("🖼️ 이미지 설명", expanded=False):
                    st.write(result['image_desc'])

            # 다운로드 버튼
            st.markdown("---")
            result_text = f"""
# 뉴스 심의문 분석 결과

## 기사 정보
- **제목:** {article.get('title', 'N/A')}
- **언론사:** {article.get('media', 'N/A')}
- **날짜:** {article.get('date', 'N/A')}
- **URL:** {article.get('url', url_input)}
- **이미지 URL:** {article.get('image_url', 'N/A')}

## 분석 결과
- **결과:** {"위반 없음" if result['final_decision'].strip() == "위반 없음" else "위반"}
- **유사 사례 위반 수:** {result['violation_count']}/5

## 최종 심의문
{result['final_decision']}

## 유사 사례
{result['similar_cases']}
"""

            st.download_button(
                label="💾 결과 다운로드 (TXT)",
                data=result_text,
                file_name=f"심의문_분석결과_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

else:
    st.info("👆 기사 URL을 입력하고 '분석 시작' 버튼을 클릭하세요.")
