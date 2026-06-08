# AI Attorney Game (역전재판 스타일)

AI 기반 법정·추리 시뮬레이션 웹 게임입니다. 플레이어는 변호사가 되어 현장 수색으로 증거를 모으고, 증인·검사와 대결하며 모순을 지적합니다. LLM이 증인·검사·판사 대사와 판정을 생성합니다.

**플레이 가능 에피소드**
- `turnabout_clock` — 첫 번째 사건 (시간을 달려서)
- `disaster_epitaph` — 두 번째 사건 (재앙의 임말소, 3차 재판)

---

## 프로젝트 구조

```
역전재판/                          # 워크스페이스 루트 (원본 아트·증거 원본)
├── ai_attorney_game/              # ★ 실행 가능한 메인 앱
│   ├── backend/                   # FastAPI 서버, AI 액터, 재판 엔진
│   │   ├── main.py                # API·WebSocket 진입점
│   │   ├── ai_services/           # 검사·증인·판사·답변평가 LLM 프롬프트
│   │   ├── core/                  # court_orchestrator, free_dialogue_engine 등
│   │   ├── services/              # episode_loader, DB
│   │   ├── schemas/               # episode/trial Pydantic 모델
│   │   └── tests/
│   ├── frontend/                  # React + Vite 클라이언트
│   │   ├── public/                # 정적 에셋 (수사·법정·스토리 이미지)
│   │   │   ├── investigation/     # 수사 씬 배경·증거 스프라이트
│   │   │   ├── court-assets/      # 법정 캐릭터·SFX
│   │   │   ├── epitaph-interstitial/
│   │   │   └── intro/
│   │   └── src/
│   │       ├── pages/             # Courtroom, Investigation, EpisodeSelect 등
│   │       ├── data/              # 스토리 흐름, 증거 경로, 재판 단축키
│   │       └── investigation/     # investigationConfig.js (수사 맵·오브젝트)
│   ├── data/
│   │   └── episodes/              # ★ 에피소드 JSON (재판 스테이지·증거 정의)
│   ├── docs/                      # 시나리오 정리 (disaster_epitaph_trials_summary.md)
│   ├── scripts/                   # 에피소드·에셋 유틸 스크립트
│   └── requirements.txt
├── 증거사진/                      # 원본 증거 이미지 (public/investigation 으로 복사해 사용)
├── 12/, 23/, 인물/, 배경 및 장식품/  # 원본 아트·컷씬 소스
└── 두번째사건설명/                # 재앙의 임말소 스토리 컷 원본
```

### 핵심 파일 위치

| 용도 | 경로 |
|------|------|
| 에피소드·재판·증거 정의 | `data/episodes/disaster_epitaph.json`, `turnabout_clock.json` |
| 수사 맵·오브젝트 배치 | `frontend/src/investigation/investigationConfig.js` |
| 증거 이미지 URL 매핑 | `frontend/src/data/evidenceAssets.js` |
| 스토리 컷·내러티브 흐름 | `frontend/src/data/epitaphStoryFlow.js` |
| 재판별 증거 필터·단축키 | `frontend/src/data/disasterEpitaphTrials.js` |
| 검사 LLM 프롬프트 | `backend/ai_services/prosecutor_actor.py` |
| 증인 LLM 프롬프트 | `backend/ai_services/witness_actor.py` |
| 판사 LLM 프롬프트 | `backend/ai_services/judge_actor.py` |
| 답변 판정(모순 인정 여부) | `backend/ai_services/answer_evaluator.py` |
| 재판 오케스트레이션 | `backend/core/court_orchestrator.py`, `free_dialogue_engine.py` |

> **중요:** `data/episodes/*.json`을 수정한 뒤에는 **백엔드를 재시작**해야 변경이 반영됩니다. (에피소드는 서버 기동 시 로드됩니다.)

---

## 사전 요구사항

- **Python** 3.11+ (3.14 테스트됨)
- **Node.js** 18+ 및 **npm**
- (선택) OpenAI API 키 — 없으면 mock 모드로 동작

---

## 실행 방법


### 1. Backend

```powershell
cd ai_attorney_game
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```


### 2. Frontend

```powershell
cd ai_attorney_game/frontend
npm install
npm run dev
```

### 3. 환경 변수

`ai_attorney_game/frontend/.env` (또는 `.env.example` 참고):

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_WS_BASE_URL=ws://127.0.0.1:8000
```

백엔드 루트 `ai_attorney_game/.env`:

```env
USE_REDIS=false
OPENAI_API_KEY=
OPENAI_PROSECUTOR_API_KEY=
OPENAI_WITNESS_API_KEY=
OPENAI_JUDGE_API_KEY=
OPENAI_SYSTEM_API_KEY=
OPENAI_MODEL=gpt-4o-mini
LLM_PROVIDER=auto
```

- `USE_REDIS=false` → Redis 없이 인메모리 세션
- 역할별 키가 비어 있으면 `OPENAI_API_KEY`를 fallback으로 사용
- API 키 없거나 실패 시 **mock 모드** 자동 fallback
- mock만 쓰려면 `LLM_PROVIDER=mock`

### 4. 접속

[http://localhost:5173](http://localhost:5173)

---

## 개발 단축키 (disaster_epitaph, DEV)

| 단축키 | 동작 |
|--------|------|
| `1` → `Enter` | 1차 재판 전 스토리로 스킵 (이후 클럽 수사) |
| `2` → `Enter` | 1차 재판 후 스토리로 스킵 (이후 차고 수사) |
| `3` → `Enter` | (에피소드 선택 화면) 3차 재판 스토리 구간 — `resolveStoryShortcutPhase`에 정의된 경우 |
| `Shift` + `Enter` | 수사 화면에서 **법정 바로 진입** (증거 자동 수집) |

- 숫자 키 입력 후 2초 이내 Enter를 눌러야 합니다.
- 사건 선택 화면·수사 화면 모두에서 `1`/`2`/`3` + Enter 동작합니다.

**수사 조작:** 방향키 이동, `E` 증거 목록, `M` 장소 선택

**개발 URL:** `?devInvestigation=disaster_epitaph&phase=garage` — 수사 씬 직접 진입

---

## disaster_epitaph (재앙의 임말소) 플레이 흐름

1. **인트로** → 사건 선택 → 난이도 선택
2. **스토리** (전쟁·YJ그룹 붕괴·클럽 사건 배경)
3. **1차 수사** — 클럽 VX 관련 증거 수집
4. **1차 재판** (`trial_epitaph_1`) — 피고 소호, 증인 이소은  
   - 핵심: VX 피부 치사량 10mg vs 「손에 쏟고 춤」 → **이소은 자백**
5. **1차 재판 후 스토리** (이소은 연행·차량 사고)
6. **2차 수사** — 차고 (도청기, 앤서니 신분증, 차량 CCTV 등)
7. **2차 재판** (`trial_epitaph_2`) — 피고 앤서니  
   - 핵심: 서버 로그 좌회전 vs CCTV 우회전, 동기≠유죄 → **재판 연기**
8. **3차 재판** (`trial_epitaph_3`) — 증인 임민수  
   - 핵심: 얼굴 기록 조작(그림자), 유언장 뒷장 함정 → **임민수 자백, 앤서니 무죄**

상세 모순·증거 표는 `docs/disaster_epitaph_trials_summary.md` 참고.

---

## 플레이 구조 (공통)

1. 현장 수색 → 수사 노트 증거가 법정 인벤토리에 동기화
2. 법정 `vs_witness` 스테이지에서 증언 공격
3. 증거 또는 발언 기록 최대 2개 + 100자 이내 자유 주장
4. StageEngine 판정 → 판사 설명 → 증인 반응 → (필요 시) 검사 개입
5. 증인 멘탈 0 → 스테이지 클리어

---

## 테스트

```powershell
cd ai_attorney_game
pytest backend/tests -q
```

---

## 에셋 복사 참고

원본 증거 이미지는 워크스페이스 루트 `증거사진/`에 있으며, 게임에서 쓰는 경로는 `frontend/public/investigation/`입니다.  
예: `증거사진/차량충돌모습.png` → `frontend/public/investigation/차량충돌모습.png`

---

*다른 에이전트·개발자용 요약 문서 — 프로젝트 루트 `역전재판_export_*.zip`과 함께 배포*
