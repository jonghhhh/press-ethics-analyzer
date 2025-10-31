# 📰 뉴스 심의문 분석 시스템

한국신문윤리위원회 심의 기준을 적용한 AI 기반 뉴스 기사 윤리 분석 시스템입니다.

## 🌟 주요 기능

- **기사 자동 추출**: URL만 입력하면 기사 내용 자동 수집
- **멀티모달 분석**: 텍스트와 이미지를 함께 분석 (Gemini 2.0 Flash)
- **유사 사례 검색**: ChromaDB 기반 벡터 검색으로 관련 심의 사례 탐색
- **심의문 자동 생성**: 신문윤리실천요강 16개 조항 기준 자동 판단
- **단계별 진행 표시**: 5단계 분석 과정 실시간 모니터링

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/jonghhhh/press-ethics-analyzer.git
cd press-ethics-analyzer
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. ChromaDB 데이터 다운로드 (필수)

**중요**: 데이터베이스 파일은 GitHub에 포함되지 않습니다. Hugging Face에서 다운로드해야 합니다.

```bash
# Hugging Face Space 클론
git clone https://huggingface.co/spaces/jonghhhh/press_ethics temp_hf

# 데이터베이스 파일 복사
cp temp_hf/chroma/chroma.sqlite3 ./chroma/
cp -r temp_hf/chroma/4f0d0b85-04bb-4afc-9e18-51530c937a3b ./chroma/

# 임시 디렉토리 삭제
rm -rf temp_hf
```

자세한 내용은 [chroma/README.md](chroma/README.md)를 참조하세요.

## 🔑 API 키 설정

### Gemini API 키 발급

1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. API 키 생성
3. Streamlit 앱 사이드바에서 API 키 입력

또는 `.env` 파일 생성:

```bash
GEMINI_KEY=your_gemini_api_key_here
```

## 💻 실행 방법

### Streamlit 앱 실행

```bash
streamlit run multimodal_rag_langgraph_gemini_st.py
```

브라우저에서 `http://localhost:8501` 자동 실행

### CLI 버전 실행

```bash
python multimodal_rag_langgraph_gemini.py
```

## 📋 사용 방법

1. **API 키 입력**: 사이드바에서 Gemini API 키 입력
2. **URL 입력**: 분석할 뉴스 기사 URL 입력
3. **분석 시작**: '분석 시작' 버튼 클릭
4. **결과 확인**: 5단계 진행 과정을 거쳐 최종 심의문 확인

## 🏗️ 프로젝트 구조

```
press-ethics-analyzer/
├── multimodal_rag_langgraph_gemini.py      # CLI 버전
├── multimodal_rag_langgraph_gemini_st.py   # Streamlit 웹앱
├── multimodal_rag_langgraph.py             # Ollama 버전 (로컬)
├── analyze_excel_articles.py               # 배치 분석 스크립트
├── news_text_scraper.py                    # 뉴스 스크래핑 모듈
├── requirements.txt                        # 패키지 의존성
├── chroma/                                 # ChromaDB 데이터
└── README.md                               # 문서
```

## 🔍 분석 단계

1. **기사 추출**: URL에서 제목, 본문, 이미지 추출
2. **이미지 처리**: Gemini 멀티모달로 이미지 내용 분석
3. **유사 사례 검색**: 벡터 DB에서 관련 심의 사례 5개 검색
4. **심의문 생성**: LangGraph 워크플로우로 심의문 초안 작성
5. **최종 검토**: 조항 정확성 및 기사 관련성 검증

## 📊 심의 기준

### 신문윤리실천요강 16개 조항

- 제1조: 언론의 자유, 책임, 독립
- 제2조: 취재 준칙
- 제3조: 보도 준칙
- 제4조: 사법 보도 준칙
- 제5조: 취재원의 명시와 보호
- 제6조: 보도유예 시한
- 제7조: 범죄보도와 인권존중
- 제8조: 저작물의 전재와 인용
- 제9조: 평론의 원칙
- 제10조: 편집 지침
- 제11조: 명예와 신용존중
- 제12조: 사생활 보호
- 제13조: 청소년과 어린이 보호
- 제14조: 정보의 부당이용금지
- 제15조: 언론인의 품위
- 제16조: 공익의 정의

## 🛠️ 기술 스택

- **AI 모델**: Google Gemini 2.0 Flash
- **프레임워크**: LangGraph, Streamlit
- **벡터 DB**: ChromaDB
- **임베딩**: Sentence Transformers (multilingual-e5-large-instruct)
- **스크래핑**: BeautifulSoup4, Requests
- **이미지 처리**: Pillow

## 🚀 클라우드 배포

### Render.com 배포

1. [Render.com](https://render.com/) 가입
2. "New +" → "Web Service" 선택
3. GitHub 저장소 연결: `jonghhhh/press-ethics-analyzer`
4. 자동 감지된 설정 확인 (render.yaml 사용)
5. "Create Web Service" 클릭

**참고**: 초기 빌드 시 ChromaDB 파일(225MB)을 Hugging Face에서 자동 다운로드하므로 5-10분 소요됩니다.

### Hugging Face Spaces (권장)

이미 배포된 버전을 바로 사용할 수 있습니다:
- 🌐 [https://huggingface.co/spaces/jonghhhh/press_ethics](https://huggingface.co/spaces/jonghhhh/press_ethics)

### 기타 플랫폼

- **Streamlit Cloud**: GitHub 저장소 연결 후 `app.py` 지정
- **Railway**: GitHub 연결 시 자동 감지
- **Fly.io**: Dockerfile 생성 필요

## ⚠️ 주의사항

- API 키는 절대 공개 저장소에 커밋하지 마세요
- ChromaDB 데이터는 사전에 준비되어야 합니다
- 분석에는 수 분이 소요될 수 있습니다
- GPU를 사용하려면 PyTorch GPU 버전을 별도 설치하세요
- 클라우드 배포 시 초기 빌드는 모델 다운로드로 인해 시간이 걸립니다

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.

## 🤝 기여

이슈 제보와 PR은 언제나 환영합니다!

## 📧 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
