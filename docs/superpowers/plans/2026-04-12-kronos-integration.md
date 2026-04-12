# Kronos Foundation Model 통합 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** daily-stock-report와 stock-analyzer 스킬에 Kronos-mini 확률적 가격 예측을 통합한다.

**Architecture:** 공유 래퍼(`libs/kronos_predictor.py`)가 Kronos-mini 모델을 로드하고, yfinance OHLCV 데이터를 받아 Monte Carlo 샘플링으로 1일/5일 예측 분포를 생성한다. 리포트(`ai_forecast.py`)와 스킬(`main.py`) 양쪽에서 이 래퍼를 참조한다.

**Tech Stack:** Python 3.12, PyTorch (CPU), Kronos-mini (4.1M params, HuggingFace), yfinance, pandas

**Kronos API 핵심 제약:** `sample_count > 1`은 내부에서 평균을 반환하므로 개별 경로 접근 불가. p10/p90 분포를 얻으려면 `sample_count=1`로 N번 독립 호출 후 통계를 계산해야 한다.

---

## 파일 구조

| 파일 | 작업 | 역할 |
|------|------|------|
| `libs/kronos_predictor.py` | NEW | Kronos 래퍼 — predict(), predict_batch() |
| `scripts/sort_utils.py` | NEW | 시장/시총 기반 정렬 유틸리티 |
| `scripts/ai_forecast.py` | NEW | 배치 예측 → HTML [AI 예측] 섹션 |
| `scripts/report_generator.py` | MODIFY | 정렬 + ai_forecast 호출 추가 |
| `scripts/analyzer.py` | NO CHANGE | 정렬은 report_generator.py에서 처리, analyzer는 변경 불필요 |
| `scripts/extras.py` | NO CHANGE | 이미 리스트를 수신하므로 변경 불필요 |
| `requirements.txt` | MODIFY | torch, einops, huggingface_hub, safetensors 추가 |
| `.github/workflows/daily-report.yml` | MODIFY | 타임아웃 20분, torch CPU 설치 |
| `tests/test_kronos_predictor.py` | NEW | 래퍼 단위 테스트 |
| `tests/test_sort_utils.py` | NEW | 정렬 단위 테스트 |
| `tests/test_ai_forecast.py` | NEW | HTML 섹션 생성 테스트 |
| `~/.claude/skills/stock-analyzer/scripts/main.py` | MODIFY | forecast 키 추가 |
| `~/.claude/skills/stock-analyzer/SKILL.md` | MODIFY | Claude 해석 가이드 추가 |

---

### Task 1: Kronos 레포 클론 및 의존성 설정

**Files:**
- Modify: `requirements.txt`
- Modify: `.github/workflows/daily-report.yml`

- [ ] **Step 1: Kronos 레포를 서브디렉토리로 클론**

```bash
cd /Users/miki/Developer/daily-stock-report
git clone https://github.com/shiyu-coder/Kronos.git libs/kronos-repo
```

이 디렉토리는 `.gitignore`에 추가하여 추적하지 않는다 (런타임 의존성).

- [ ] **Step 2: .gitignore에 kronos-repo 추가**

```bash
echo "libs/kronos-repo/" >> .gitignore
```

- [ ] **Step 3: requirements.txt에 의존성 추가**

기존 `requirements.txt` 끝에 추가:

```
# Kronos Foundation Model
torch>=2.0.0
einops>=0.8.1
huggingface_hub>=0.33.1
safetensors>=0.6.2
```

- [ ] **Step 4: GitHub Actions workflow 수정**

`.github/workflows/daily-report.yml` 전체 내용:

```yaml
name: Daily Stock Report

on:
  schedule:
    - cron: '0 23 * * 0-4'
    - cron: '30 5 * * 1-5'
  workflow_dispatch:

jobs:
  report:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt
      - name: Clone Kronos model code
        run: git clone --depth 1 https://github.com/shiyu-coder/Kronos.git libs/kronos-repo
      - name: Generate and send report
        env:
          GMAIL_USER: ${{ secrets.GMAIL_USER }}
          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
          REPORT_EMAIL: ${{ secrets.REPORT_EMAIL }}
        working-directory: scripts
        run: |
          python report_generator.py ../watchlist.json | python send_email.py
```

- [ ] **Step 5: 로컬에서 의존성 설치 확인**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops huggingface_hub safetensors
```

Expected: 모두 설치 성공, torch CPU 버전 (~200MB)

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt .github/workflows/daily-report.yml .gitignore
git commit -m "chore: add Kronos dependencies and update workflow timeout"
```

---

### Task 2: kronos_predictor.py 래퍼 구현

**Files:**
- Create: `libs/__init__.py`
- Create: `libs/kronos_predictor.py`
- Create: `tests/test_kronos_predictor.py`

- [ ] **Step 1: 테스트 파일 작성**

`tests/test_kronos_predictor.py`:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs'))


class TestKronosPredictor:
    """KronosPredictor 래퍼 테스트. 실제 모델 로딩 없이 mock으로 테스트."""

    def _make_ohlcv_df(self, days=90):
        """테스트용 OHLCV DataFrame 생성."""
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        np.random.seed(42)
        close = 50000 + np.cumsum(np.random.randn(days) * 500)
        return pd.DataFrame({
            'Open': close - np.random.rand(days) * 200,
            'High': close + np.random.rand(days) * 300,
            'Low': close - np.random.rand(days) * 300,
            'Close': close,
            'Volume': np.random.randint(1000000, 5000000, days),
        }, index=dates)

    def test_format_yfinance_to_kronos(self):
        """yfinance DataFrame → Kronos 입력 변환 테스트."""
        from kronos_predictor import KronosPredictor

        df = self._make_ohlcv_df(90)
        predictor = KronosPredictor.__new__(KronosPredictor)
        x_df, x_ts, y_ts = predictor._prepare_input(df, pred_len=5)

        assert list(x_df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert len(x_df) == 90
        assert len(y_ts) == 5
        assert x_df['close'].iloc[-1] == df['Close'].iloc[-1]

    def test_compute_statistics(self):
        """Monte Carlo 샘플에서 통계 계산 테스트."""
        from kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        # 5개 샘플, 5일 예측, close 값만
        samples = [
            [100, 102, 104, 106, 108],
            [100, 98, 96, 94, 92],
            [100, 101, 102, 103, 104],
            [100, 103, 106, 109, 112],
            [100, 99, 98, 97, 96],
        ]
        current_price = 100.0

        result = predictor._compute_statistics(samples, current_price, horizons=[1, 5])

        assert "1d" in result
        assert "5d" in result
        # 1일 예측: 샘플들의 day-0 값 = [100,100,100,100,100] → median=100
        # 5일 예측: 샘플들의 day-4 값 = [108,92,104,112,96] → median=104
        assert result["5d"]["median"] == pytest.approx(104.0)
        assert 0.0 <= result["1d"]["direction_prob"] <= 1.0
        assert result["1d"]["p10"] <= result["1d"]["median"] <= result["1d"]["p90"]

    def test_predict_batch_timeout(self):
        """predict_batch 타임아웃 시 fallback 종목 우선 처리 테스트."""
        from kronos_predictor import KronosPredictor
        import time

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor.model = None
        predictor.tokenizer = None
        predictor.predictor = None

        call_order = []

        def mock_predict(df, **kwargs):
            call_order.append(df.attrs.get('ticker', 'unknown'))
            time.sleep(0.01)
            return {"1d": {"median": 100}, "5d": {"median": 100}}

        predictor.predict = mock_predict

        df1 = self._make_ohlcv_df(90)
        df1.attrs['ticker'] = 'NVDA'
        df2 = self._make_ohlcv_df(90)
        df2.attrs['ticker'] = 'OTHER'
        df3 = self._make_ohlcv_df(90)
        df3.attrs['ticker'] = 'AAPL'

        tickers_data = {'NVDA': df1, 'OTHER': df2, 'AAPL': df3}
        fallback = {'NVDA', 'AAPL'}

        results = predictor.predict_batch(
            tickers_data, timeout_seconds=10, fallback_tickers=fallback
        )

        # fallback 종목이 먼저 처리되어야 함
        assert call_order[0] in fallback
        assert call_order[1] in fallback
        assert len(results) == 3

    def test_predict_returns_none_on_insufficient_data(self):
        """60일 미만 데이터 시 None 반환 테스트."""
        from kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor.model = None
        predictor.tokenizer = None
        predictor.predictor = None

        df = self._make_ohlcv_df(30)  # 60일 미만
        result = predictor.predict(df)
        assert result is None
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
cd /Users/miki/Developer/daily-stock-report
python -m pytest tests/test_kronos_predictor.py -v
```

Expected: FAIL — `kronos_predictor` 모듈 없음

- [ ] **Step 3: libs/__init__.py 생성**

```bash
touch libs/__init__.py
```

- [ ] **Step 4: kronos_predictor.py 구현**

`libs/kronos_predictor.py`:

```python
"""Kronos Foundation Model 래퍼.

daily-stock-report와 stock-analyzer 스킬 양쪽에서 사용하는
확률적 가격 예측 공유 모듈.

Kronos-mini (4.1M params)를 CPU에서 실행하며,
Monte Carlo 샘플링으로 1일/5일 예측 분포를 생성한다.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kronos 레포 경로 (libs/kronos-repo/)
_KRONOS_REPO = os.path.join(os.path.dirname(__file__), "kronos-repo")
if os.path.isdir(_KRONOS_REPO) and _KRONOS_REPO not in sys.path:
    sys.path.insert(0, _KRONOS_REPO)

_MIN_HISTORY_DAYS = 60


class KronosPredictor:
    """Kronos-mini 확률적 가격 예측기."""

    def __init__(self, model_name="NeoQuasar/Kronos-mini",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k",
                 device="cpu"):
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor as _KP

            self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
            self.model = Kronos.from_pretrained(model_name)
            self.predictor = _KP(
                self.model, self.tokenizer,
                device=device, max_context=2048
            )
            self._available = True
            logger.info("Kronos-mini loaded on %s", device)
        except Exception as e:
            logger.warning("Kronos 로드 실패, 예측 비활성화: %s", e)
            self._available = False
            self.predictor = None

    def predict(self, df, horizons=None, n_samples=5):
        """단일 종목 확률적 예측.

        Args:
            df: yfinance 형식 DataFrame (Open, High, Low, Close, Volume).
                최소 60일 이상.
            horizons: 추출할 예측 일수 리스트. 기본 [1, 5].
            n_samples: Monte Carlo 샘플 수. 기본 5.

        Returns:
            dict {"1d": {...}, "5d": {...}} 또는 None (데이터 부족/에러).
        """
        if horizons is None:
            horizons = [1, 5]

        if df is None or len(df) < _MIN_HISTORY_DAYS:
            return None

        if not self._available:
            return None

        max_horizon = max(horizons)

        try:
            x_df, x_ts, y_ts = self._prepare_input(df, pred_len=max_horizon)
        except Exception as e:
            logger.warning("입력 변환 실패: %s", e)
            return None

        close_samples = []
        for _ in range(n_samples):
            try:
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=max_horizon,
                    T=1.0,
                    top_k=0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False,
                )
                close_samples.append(pred_df['close'].tolist())
            except Exception as e:
                logger.warning("예측 샘플 실패: %s", e)
                continue

        if len(close_samples) < 2:
            return None

        current_price = float(df['Close'].iloc[-1])
        return self._compute_statistics(close_samples, current_price, horizons)

    def predict_batch(self, tickers_data, horizons=None, n_samples=5,
                      timeout_seconds=600, fallback_tickers=None):
        """다종목 배치 예측 (타임아웃 지원).

        Args:
            tickers_data: {ticker: DataFrame} 딕셔너리.
            horizons: 예측 일수 리스트. 기본 [1, 5].
            n_samples: Monte Carlo 샘플 수.
            timeout_seconds: 전체 타임아웃 (초).
            fallback_tickers: 우선 처리할 종목 set.

        Returns:
            {ticker: predict() 결과 또는 None} 딕셔너리.
        """
        if horizons is None:
            horizons = [1, 5]
        if fallback_tickers is None:
            fallback_tickers = set()

        results = {}
        start = time.time()
        cutoff = timeout_seconds * 0.7

        # Phase 1: fallback 종목 우선
        priority = [t for t in tickers_data if t in fallback_tickers]
        rest = [t for t in tickers_data if t not in fallback_tickers]
        ordered = priority + rest

        for ticker in ordered:
            elapsed = time.time() - start
            if elapsed > cutoff:
                logger.info("타임아웃 근접 (%.0fs/%.0fs), 나머지 %d종목 스킵",
                            elapsed, timeout_seconds,
                            len(ordered) - len(results))
                for remaining in ordered:
                    if remaining not in results:
                        results[remaining] = None
                break

            df = tickers_data[ticker]
            try:
                results[ticker] = self.predict(df, horizons=horizons,
                                               n_samples=n_samples)
            except Exception as e:
                logger.warning("%s 예측 실패: %s", ticker, e)
                results[ticker] = None

        return results

    def _prepare_input(self, df, pred_len=5):
        """yfinance DataFrame → Kronos 입력 형식 변환.

        Returns:
            (x_df, x_timestamp, y_timestamp) 튜플.
        """
        x_df = pd.DataFrame({
            'open': df['Open'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'close': df['Close'].values,
            'volume': df['Volume'].values.astype(float),
        })

        # 타임스탬프 생성
        if isinstance(df.index, pd.DatetimeIndex):
            x_ts = pd.Series(df.index)
        else:
            x_ts = pd.Series(pd.bdate_range(
                end=pd.Timestamp.today(), periods=len(df)
            ))

        last_date = x_ts.iloc[-1]
        y_ts = pd.Series(pd.bdate_range(
            start=last_date + pd.offsets.BDay(1), periods=pred_len
        ))

        return x_df, x_ts, y_ts

    def _compute_statistics(self, samples, current_price, horizons):
        """Monte Carlo 샘플 리스트에서 통계 계산.

        Args:
            samples: list of list[float] — 각 샘플의 일별 close 예측값.
            current_price: 현재가.
            horizons: [1, 5] 등.

        Returns:
            {"1d": {median, p10, p90, direction_prob, volatility}, ...}
        """
        arr = np.array(samples)  # shape: (n_samples, max_horizon)
        result = {}

        for h in horizons:
            idx = h - 1  # 0-indexed
            if idx >= arr.shape[1]:
                continue

            values = arr[:, idx]
            median = float(np.median(values))
            p10 = float(np.percentile(values, 10))
            p90 = float(np.percentile(values, 90))
            up_count = np.sum(values > current_price)
            direction_prob = float(up_count / len(values))

            # 일간 변동성: 현재가 대비 표준편차 비율
            volatility = float(np.std(values) / current_price) if current_price > 0 else 0.0

            result[f"{h}d"] = {
                "median": round(median, 2),
                "p10": round(p10, 2),
                "p90": round(p90, 2),
                "direction_prob": round(direction_prob, 2),
                "volatility": round(volatility, 4),
            }

        return result
```

- [ ] **Step 5: 테스트 실행**

```bash
python -m pytest tests/test_kronos_predictor.py -v
```

Expected: 4 tests PASS

- [ ] **Step 6: 커밋**

```bash
git add libs/ tests/test_kronos_predictor.py
git commit -m "feat: add KronosPredictor wrapper with Monte Carlo sampling"
```

---

### Task 3: sort_utils.py 정렬 유틸리티

**Files:**
- Create: `scripts/sort_utils.py`
- Create: `tests/test_sort_utils.py`

- [ ] **Step 1: 테스트 파일 작성**

`tests/test_sort_utils.py`:

```python
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from sort_utils import sort_by_market_and_cap


class TestSortByMarketAndCap:

    def test_krx_before_us(self):
        tickers = [
            {"ticker": "AAPL", "name": "Apple", "market": "US"},
            {"ticker": "005930.KS", "name": "삼성전자", "market": "KRX"},
        ]
        cache = {
            "AAPL": {"marketCap": 3000000000000},
            "005930.KS": {"marketCap": 400000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "005930.KS"
        assert result[1]["ticker"] == "AAPL"

    def test_sort_by_cap_within_group(self):
        tickers = [
            {"ticker": "035720.KS", "name": "카카오", "market": "KRX"},
            {"ticker": "005930.KS", "name": "삼성전자", "market": "KRX"},
            {"ticker": "035420.KS", "name": "네이버", "market": "KRX"},
        ]
        cache = {
            "035720.KS": {"marketCap": 20000000000000},
            "005930.KS": {"marketCap": 400000000000000},
            "035420.KS": {"marketCap": 50000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "005930.KS"
        assert result[1]["ticker"] == "035420.KS"
        assert result[2]["ticker"] == "035720.KS"

    def test_missing_cap_goes_last(self):
        tickers = [
            {"ticker": "FIGM", "name": "Figma", "market": "US"},
            {"ticker": "AAPL", "name": "Apple", "market": "US"},
        ]
        cache = {
            "AAPL": {"marketCap": 3000000000000},
            # FIGM: 미상장, marketCap 없음
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "AAPL"
        assert result[1]["ticker"] == "FIGM"

    def test_kosdaq_grouped_with_kospi(self):
        tickers = [
            {"ticker": "MSFT", "name": "Microsoft", "market": "US"},
            {"ticker": "035900.KQ", "name": "JYP", "market": "KRX"},
        ]
        cache = {
            "MSFT": {"marketCap": 3000000000000},
            "035900.KQ": {"marketCap": 5000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "035900.KQ"  # KRX 그룹이 먼저
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
python -m pytest tests/test_sort_utils.py -v
```

Expected: FAIL — `sort_utils` 모듈 없음

- [ ] **Step 3: sort_utils.py 구현**

`scripts/sort_utils.py`:

```python
"""시장/시총 기반 워치리스트 정렬 유틸리티."""

import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# KRX 티커 접미사
_KRX_SUFFIXES = ('.KS', '.KQ')


def fetch_market_caps(tickers):
    """워치리스트의 시가총액을 일괄 조회.

    Args:
        tickers: [{"ticker": "005930.KS", ...}, ...]

    Returns:
        {ticker_str: {"marketCap": int}} 딕셔너리. 조회 실패 시 해당 키 없음.
    """
    cache = {}
    for stock in tickers:
        ticker = stock["ticker"]
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap")
            if cap and cap > 0:
                cache[ticker] = {"marketCap": cap}
        except Exception as e:
            logger.warning("시총 조회 실패 %s: %s", ticker, e)
    return cache


def sort_by_market_and_cap(tickers, ticker_info_cache=None):
    """시장 그룹 (KRX → US) + 시총 내림차순 정렬.

    Args:
        tickers: [{"ticker": "005930.KS", "name": "삼성전자", "market": "KRX"}, ...]
        ticker_info_cache: {ticker: {"marketCap": int}}. None이면 yfinance 조회.

    Returns:
        정렬된 리스트 (새 리스트, 원본 불변).
    """
    if ticker_info_cache is None:
        ticker_info_cache = fetch_market_caps(tickers)

    def sort_key(stock):
        ticker = stock["ticker"]
        # 1차: KRX(0) → US(1)
        is_krx = 0 if ticker.endswith(_KRX_SUFFIXES) else 1
        # 2차: marketCap 내림차순 (없으면 0 → 맨 뒤)
        cap = ticker_info_cache.get(ticker, {}).get("marketCap", 0)
        return (is_krx, -cap)

    return sorted(tickers, key=sort_key)
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/test_sort_utils.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/sort_utils.py tests/test_sort_utils.py
git commit -m "feat: add sort_utils for market/cap-based watchlist sorting"
```

---

### Task 4: report_generator.py에 정렬 적용

**Files:**
- Modify: `scripts/report_generator.py`

- [ ] **Step 1: report_generator.py 상단에 import 추가**

`scripts/report_generator.py` 상단 import 영역에:

```python
from sort_utils import sort_by_market_and_cap, fetch_market_caps
```

- [ ] **Step 2: analyze_watchlist 호출 후 정렬 로직 삽입**

현재 코드 (약 line 77-80):

```python
results = analyze_watchlist(watchlist_path)
krx_stocks = [r for r in results if r['market'] == 'KRX']
us_stocks  = [r for r in results if r['market'] == 'US']
```

변경:

```python
results = analyze_watchlist(watchlist_path)

# 시총 기반 정렬
ticker_info_cache = fetch_market_caps(results)
results = sort_by_market_and_cap(results, ticker_info_cache)

krx_stocks = [r for r in results if r['market'] == 'KRX']
us_stocks  = [r for r in results if r['market'] == 'US']
```

`results`의 각 dict에는 `ticker`, `name`, `market` 키가 있으므로 `sort_by_market_and_cap`이 그대로 동작한다. 이미 KRX/US로 분리하기 전에 정렬하므로, 각 그룹 내에서도 시총 순서가 유지된다.

- [ ] **Step 3: 로컬 테스트**

```bash
cd scripts
python report_generator.py ../watchlist.json > /tmp/report_sorted.html
open /tmp/report_sorted.html
```

Expected: HTML에서 한국 주식이 시총 순으로, 미국 주식이 시총 순으로 표시.

- [ ] **Step 4: 커밋**

```bash
git add scripts/report_generator.py
git commit -m "feat: sort watchlist by market group and market cap"
```

---

### Task 5: ai_forecast.py HTML 섹션 생성

**Files:**
- Create: `scripts/ai_forecast.py`
- Create: `tests/test_ai_forecast.py`

- [ ] **Step 1: 테스트 파일 작성**

`tests/test_ai_forecast.py`:

```python
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from ai_forecast import _format_price, _direction_icon, _volatility_label, build_forecast_html


class TestFormatHelpers:

    def test_format_price_krw(self):
        assert _format_price(58500, "KRW") == "₩58,500"

    def test_format_price_usd(self):
        assert _format_price(138.25, "USD") == "$138.25"

    def test_direction_icon_up(self):
        assert _direction_icon(0.62) == "🔼"

    def test_direction_icon_down(self):
        assert _direction_icon(0.38) == "🔽"

    def test_direction_icon_sideways(self):
        assert _direction_icon(0.50) == "➡️"

    def test_volatility_label(self):
        assert _volatility_label(0.035) == "높음"
        assert _volatility_label(0.02) == "보통"
        assert _volatility_label(0.01) == "낮음"


class TestBuildForecastHtml:

    def test_returns_html_with_forecasts(self):
        forecasts = {
            "005930.KS": {
                "1d": {"median": 58500, "p10": 57800, "p90": 59200,
                       "direction_prob": 0.62, "volatility": 0.018},
                "5d": {"median": 59000, "p10": 56500, "p90": 61500,
                       "direction_prob": 0.58, "volatility": 0.032},
            },
        }
        stocks = [{"ticker": "005930.KS", "name": "삼성전자",
                    "market": "KRX", "price": 58000, "currency": "KRW"}]

        html = build_forecast_html(forecasts, stocks)

        assert "AI 예측" in html
        assert "삼성전자" in html
        assert "🔼" in html
        assert "참고용" in html

    def test_skipped_ticker_shown(self):
        forecasts = {"005930.KS": None}
        stocks = [{"ticker": "005930.KS", "name": "삼성전자",
                    "market": "KRX", "price": 58000, "currency": "KRW"}]

        html = build_forecast_html(forecasts, stocks)
        assert "스킵" in html

    def test_empty_forecasts_returns_empty(self):
        html = build_forecast_html({}, [])
        assert html == ""
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
python -m pytest tests/test_ai_forecast.py -v
```

Expected: FAIL — `ai_forecast` 모듈 없음

- [ ] **Step 3: ai_forecast.py 구현**

`scripts/ai_forecast.py`:

```python
"""배치 Kronos 예측 → HTML [AI 예측] 섹션 생성."""

import sys
import os
import logging
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs'))

logger = logging.getLogger(__name__)

FALLBACK_TICKERS = {
    "005930.KS", "000660.KS", "005380.KS", "051910.KS", "373220.KS",
    "006400.KS", "035420.KS", "035720.KS", "017670.KS", "259960.KS",
    "NVDA", "AAPL", "MSFT", "GOOGL", "TSLA",
}


def _format_price(price, currency):
    if currency == "KRW":
        return f"₩{int(price):,}"
    return f"${price:,.2f}"


def _format_range(p10, p90, currency):
    if currency == "KRW":
        return f"{int(p10):,}~{int(p90):,}"
    return f"${p10:,.1f}~${p90:,.1f}"


def _direction_icon(prob):
    if prob >= 0.55:
        return "🔼"
    if prob <= 0.45:
        return "🔽"
    return "➡️"


def _volatility_label(vol):
    if vol > 0.03:
        return "높음"
    if vol >= 0.015:
        return "보통"
    return "낮음"


def generate_ai_forecast_section(sorted_watchlist, ticker_info_cache=None,
                                 timeout_seconds=600):
    """전종목 Kronos 예측 실행 → HTML 섹션 반환.

    Args:
        sorted_watchlist: 정렬된 워치리스트 (analyze_watchlist 결과).
        ticker_info_cache: 시총 캐시 (미사용, 인터페이스 통일용).
        timeout_seconds: predict_batch 타임아웃.

    Returns:
        HTML 문자열. Kronos 로드 실패 시 빈 문자열.
    """
    try:
        from kronos_predictor import KronosPredictor
        predictor = KronosPredictor()
        if not predictor._available:
            logger.warning("Kronos 비활성화, AI 예측 섹션 스킵")
            return ""
    except Exception as e:
        logger.warning("Kronos import 실패: %s", e)
        return ""

    # OHLCV 수집
    tickers_data = {}
    for stock in sorted_watchlist:
        ticker = stock['ticker']
        try:
            hist = yf.Ticker(ticker).history(period="90d")
            if len(hist) >= 60:
                tickers_data[ticker] = hist
        except Exception:
            pass

    if not tickers_data:
        return ""

    forecasts = predictor.predict_batch(
        tickers_data,
        n_samples=5,
        timeout_seconds=timeout_seconds,
        fallback_tickers=FALLBACK_TICKERS,
    )

    return build_forecast_html(forecasts, sorted_watchlist)


def build_forecast_html(forecasts, stocks):
    """예측 결과 → HTML 테이블.

    Args:
        forecasts: {ticker: predict() 결과 또는 None}.
        stocks: 정렬된 워치리스트 (name, ticker, market, price, currency 포함).

    Returns:
        HTML 문자열. 예측 없으면 빈 문자열.
    """
    if not forecasts:
        return ""

    total = len(forecasts)
    done = sum(1 for v in forecasts.values() if v is not None)
    skipped = total - done

    # 1일 예측 테이블
    rows_1d = []
    rows_5d = []

    for stock in stocks:
        ticker = stock['ticker']
        fc = forecasts.get(ticker)
        name = stock.get('name', ticker)
        price = stock.get('price', 0)
        currency = stock.get('currency', 'KRW' if ticker.endswith(('.KS', '.KQ')) else 'USD')

        if fc is None:
            rows_1d.append(f"""
                <tr style="color:#999;">
                    <td>{name}</td><td>{_format_price(price, currency)}</td>
                    <td colspan="3" style="text-align:center;">⏱ 스킵</td>
                </tr>""")
            rows_5d.append(f"""
                <tr style="color:#999;">
                    <td>{name}</td><td>{_format_price(price, currency)}</td>
                    <td colspan="3" style="text-align:center;">⏱ 스킵</td>
                </tr>""")
            continue

        if "1d" in fc:
            d = fc["1d"]
            rows_1d.append(f"""
                <tr>
                    <td>{name}</td>
                    <td>{_format_price(price, currency)}</td>
                    <td>{_format_range(d['p10'], d['p90'], currency)}</td>
                    <td>{_direction_icon(d['direction_prob'])}</td>
                    <td>{int(d['direction_prob'] * 100)}%</td>
                </tr>""")

        if "5d" in fc:
            d = fc["5d"]
            rows_5d.append(f"""
                <tr>
                    <td>{name}</td>
                    <td>{_format_price(price, currency)}</td>
                    <td>{_format_range(d['p10'], d['p90'], currency)}</td>
                    <td>{_direction_icon(d['direction_prob'])}</td>
                    <td>{_volatility_label(d['volatility'])}</td>
                </tr>""")

    table_style = """
        <style>
        .ai-table { width:100%; border-collapse:collapse; font-size:13px; }
        .ai-table th, .ai-table td { padding:6px 10px; border-bottom:1px solid #eee; text-align:right; }
        .ai-table th { background:#f8f9fa; font-weight:600; }
        .ai-table td:first-child, .ai-table th:first-child { text-align:left; }
        </style>
    """

    def make_table(headers, rows):
        header_html = "".join(f"<th>{h}</th>" for h in headers)
        return f"""
            <table class="ai-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>"""

    skip_note = f"<p style='color:#999;font-size:12px;'>⏱ {total}종목 중 {done}종목 예측 완료 ({skipped}종목 타임아웃 스킵)</p>" if skipped > 0 else ""

    return f"""
    <div style="margin:20px 0;padding:16px;background:#fff;border:1px solid #e0e0e0;border-radius:8px;">
        {table_style}
        <h2 style="margin-top:0;">📊 AI 예측 (Kronos-mini)</h2>

        <h3>▸ 1일 예측</h3>
        {make_table(["종목", "현재가", "예상 범위", "방향", "상승확률"], rows_1d)}

        <h3 style="margin-top:16px;">▸ 5일(주간) 예측</h3>
        {make_table(["종목", "현재가", "예상 범위", "방향", "변동성"], rows_5d)}

        <p style="color:#e74c3c;font-size:12px;margin-top:12px;">
            ⚠ AI 예측은 참고용이며 투자 권유가 아닙니다.</p>
        {skip_note}
    </div>
    """
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/test_ai_forecast.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/ai_forecast.py tests/test_ai_forecast.py
git commit -m "feat: add ai_forecast.py for Kronos HTML report section"
```

---

### Task 6: report_generator.py에 AI 예측 섹션 통합

**Files:**
- Modify: `scripts/report_generator.py`

- [ ] **Step 1: import 추가**

`scripts/report_generator.py` 상단에 (Task 4에서 추가한 import 근처):

```python
from ai_forecast import generate_ai_forecast_section
```

- [ ] **Step 2: HTML 조합에 ai_forecast 섹션 삽입**

현재 HTML 조합 부분 (약 line 93 이후 — 구체적 위치는 `html +=` 패턴으로 찾기) 에서 기존 분석 테이블과 extras 사이에 추가.

기존 패턴:

```python
# ... KRX 테이블 ...
# ... US 테이블 ...
# ... extras (52주, 주간, 실적) ...
```

KRX/US 테이블 생성 후, extras 전에 삽입:

```python
# AI 예측 섹션 (Kronos)
try:
    ai_section = generate_ai_forecast_section(results, ticker_info_cache,
                                               timeout_seconds=600)
    html += ai_section
except Exception as e:
    import logging
    logging.warning("AI 예측 섹션 생성 실패: %s", e)
```

- [ ] **Step 3: 로컬 테스트 (Kronos 없이)**

```bash
cd scripts
python report_generator.py ../watchlist.json > /tmp/report_with_ai.html
open /tmp/report_with_ai.html
```

Expected: Kronos 모델이 없으면 AI 예측 섹션이 빠진 채 기존 리포트가 정상 생성됨 (graceful degradation).

- [ ] **Step 4: 로컬 테스트 (Kronos 포함)**

```bash
# libs/kronos-repo가 있고 모델 다운로드 완료된 상태에서
cd scripts
python report_generator.py ../watchlist.json > /tmp/report_full.html
open /tmp/report_full.html
```

Expected: [AI 예측] 섹션이 포함된 HTML. 1일/5일 테이블, KRX→US 순, 시총 내림차순.

- [ ] **Step 5: 커밋**

```bash
git add scripts/report_generator.py
git commit -m "feat: integrate AI forecast section into report pipeline"
```

---

### Task 7: stock-analyzer 스킬에 Kronos 통합

**Files:**
- Modify: `~/.claude/skills/stock-analyzer/scripts/main.py`
- Modify: `~/.claude/skills/stock-analyzer/SKILL.md`

- [ ] **Step 1: main.py에 Kronos 예측 추가**

`~/.claude/skills/stock-analyzer/scripts/main.py`의 `get_stock_data()` 함수 내에서 result dict 반환 직전 (약 line 302 이전)에 추가:

```python
    # Kronos 확률적 예측
    forecast = None
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.expanduser("~/Developer/daily-stock-report/libs"))
        from kronos_predictor import KronosPredictor
        _predictor = KronosPredictor()
        if _predictor._available and len(output_hist) >= 60:
            forecast = _predictor.predict(output_hist, horizons=[1, 5], n_samples=10)
    except Exception:
        pass

    if forecast:
        result["forecast"] = {
            "model": "Kronos-mini",
            **forecast,
            "disclaimer": "AI 예측은 참고용이며 투자 권유가 아닙니다.",
        }
```

`output_hist`는 이미 존재하는 변수 (yfinance history DataFrame).

- [ ] **Step 2: SKILL.md에 forecast 해석 가이드 추가**

`~/.claude/skills/stock-analyzer/SKILL.md` 하단에 추가:

```markdown
## AI 예측 (Kronos-mini)

JSON 출력에 `forecast` 키가 있으면 다음 형태로 사용자에게 제시:

```
## 📈 AI 예측 (Kronos-mini)

| 기간 | 예상 범위 | 방향 | 상승 확률 | 변동성 |
|------|----------|------|---------|--------|
| 내일 | {p10} ~ {p90} | {icon} | {direction_prob}% | {vol_label} |
| 5일 | {p10} ~ {p90} | {icon} | {direction_prob}% | {vol_label} |

> ⚠ AI 예측은 참고용이며 투자 권유가 아닙니다.
```

방향 아이콘: 🔼 ≥55%, 🔽 ≤45%, ➡️ 그 사이
변동성: >0.03 높음, 0.015~0.03 보통, <0.015 낮음

`forecast` 키가 없으면 (Kronos 미설치 등) 기존 분석만 표시. 언급하지 않음.
```

- [ ] **Step 3: 스킬 테스트**

Claude Code에서:

```
> 삼성전자 분석해줘
```

Expected: 기존 기술적 분석 + AI 예측 테이블 (Kronos 설치 시). Kronos 미설치 시 기존 분석만 표시.

- [ ] **Step 4: 커밋 (dot)**

```bash
dot add ~/.claude/skills/stock-analyzer/scripts/main.py
dot add ~/.claude/skills/stock-analyzer/SKILL.md
dot commit -m "feat: integrate Kronos forecast into stock-analyzer skill"
```

---

### Task 8: 통합 테스트 및 최종 검증

**Files:** (수정 없음, 검증만)

- [ ] **Step 1: 전체 테스트 스위트 실행**

```bash
cd /Users/miki/Developer/daily-stock-report
python -m pytest tests/ -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 2: 로컬 리포트 생성 E2E 테스트**

```bash
cd scripts
python report_generator.py ../watchlist.json > /tmp/final_report.html
open /tmp/final_report.html
```

검증 포인트:
- [ ] 전체 정렬: KRX 시총 내림차순 → US 시총 내림차순
- [ ] [AI 예측] 섹션: 1일/5일 테이블 존재
- [ ] 방향 아이콘 (🔼/🔽/➡️) 정상 표시
- [ ] 스킵 종목 "⏱ 스킵" 표시 (타임아웃 발생 시)
- [ ] 기존 섹션 (시장 요약, 기술적 분석, 52주, 숨겨진 종목) 정상

- [ ] **Step 3: GitHub Actions 수동 실행**

GitHub → Actions → Daily Stock Report → Run workflow

검증 포인트:
- [ ] Kronos 클론 + 모델 다운로드 성공
- [ ] 20분 타임아웃 내 완료
- [ ] 이메일 수신 확인

- [ ] **Step 4: 최종 커밋 및 푸시**

```bash
cd /Users/miki/Developer/daily-stock-report
git push
```
