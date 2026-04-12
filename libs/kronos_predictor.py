"""
KronosPredictor — Kronos-mini 모델을 감싸는 확률적 가격 예측 래퍼.

주요 특징:
- predict()를 N회 호출하여 Monte Carlo 분포 생성 (sample_count=1 우회)
- predict_batch()는 fallback_tickers 우선 처리 + timeout 70% 경과 시 중단
- _prepare_input(): yfinance DF(대문자) → Kronos 입력(소문자 + timestamp)
- _compute_statistics(): 샘플 경로에서 horizon별 통계 계산
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kronos 리포 경로 (이 모듈 기준 상대경로 → 절대경로 변환)
_KRONOS_REPO = str(Path(__file__).parent / "kronos-repo")


class KronosPredictor:
    """Kronos-mini를 이용한 확률적 주가 예측 래퍼.

    Args:
        model_name: HuggingFace 모델 ID (기본: NeoQuasar/Kronos-mini)
        tokenizer_name: HuggingFace 토크나이저 ID (기본: NeoQuasar/Kronos-Tokenizer-2k)
        device: 추론 장치 ('cpu', 'cuda', 'mps'). None이면 자동 감지.
    """

    MIN_INPUT_ROWS: int = 60  # predict() 최소 입력 행 수
    DEFAULT_PRED_LEN: int = 5  # 예측 기간 (영업일)
    TIMEOUT_RATIO: float = 0.70  # predict_batch timeout의 이 비율 이후 중단

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-mini",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
        device: Optional[str] = None,
    ) -> None:
        self._available: bool = False
        self.predictor = None  # type: ignore[assignment]

        # kronos-repo를 sys.path에 추가
        if _KRONOS_REPO not in sys.path:
            sys.path.insert(0, _KRONOS_REPO)

        try:
            from model.kronos import Kronos, KronosTokenizer
            from model.kronos import KronosPredictor as _KP

            logger.info("Kronos 모델 로딩 중: %s", model_name)
            kronos_model = Kronos.from_pretrained(model_name)
            kronos_tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
            self.predictor = _KP(model=kronos_model, tokenizer=kronos_tokenizer, device=device)
            self._available = True
            logger.info("Kronos 모델 로딩 완료 (device=%s)", self.predictor.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kronos 모델 로딩 실패, 예측 기능 비활성화: %s", exc)
            self._available = False

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        horizons: list[int] | None = None,
        n_samples: int = 5,
    ) -> Optional[dict[int, dict[str, float]]]:
        """단일 종목에 대한 확률적 가격 예측.

        Kronos의 sample_count>1은 평균만 반환하므로, sample_count=1로 N번 호출하여
        Monte Carlo 분포를 직접 구성한다.

        Args:
            df: yfinance 형식 DataFrame (Open/High/Low/Close/Volume, DatetimeIndex).
                최소 60행 필요.
            horizons: 예측 horizon 목록 (단위: 영업일). 기본 [1, 5].
            n_samples: Monte Carlo 샘플 수. 기본 5.

        Returns:
            horizon → {median, p10, p90, direction_prob, volatility} 딕셔너리.
            에러 또는 데이터 부족 시 None.
        """
        if horizons is None:
            horizons = [1, 5]

        # 1. 데이터 검증
        if len(df) < self.MIN_INPUT_ROWS:
            logger.debug(
                "데이터 부족: %d행 < 최소 %d행. None 반환.", len(df), self.MIN_INPUT_ROWS
            )
            return None

        # 2. 모델 미가용 시 조기 종료
        if not self._available or self.predictor is None:
            logger.warning("Kronos 모델 미가용. None 반환.")
            return None

        try:
            pred_len = self.DEFAULT_PRED_LEN
            x_df, x_ts, y_ts = self._prepare_input(df, pred_len=pred_len)
            current_price = float(df["Close"].iloc[-1])

            # Monte Carlo 샘플링: sample_count=1로 N번 반복
            close_samples: list[list[float]] = []
            for i in range(n_samples):
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=pred_len,
                    T=1.0,
                    top_k=0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False,
                )
                close_samples.append(list(pred_df["close"].values))

            if not close_samples:
                logger.warning("Monte Carlo 샘플 수집 실패: 빈 리스트. None 반환.")
                return None

            return self._compute_statistics(close_samples, current_price, horizons)

        except Exception as exc:  # noqa: BLE001
            logger.error("predict() 실패: %s", exc, exc_info=True)
            return None

    def predict_batch(
        self,
        tickers_data: dict[str, pd.DataFrame],
        horizons: list[int],
        n_samples: int,
        timeout_seconds: float = 600.0,
        fallback_tickers: list[str] | None = None,
    ) -> dict[str, Optional[dict[int, dict[str, float]]]]:
        """여러 종목 일괄 예측.

        fallback_tickers를 먼저 처리한 뒤 나머지 종목을 처리한다.
        timeout_seconds의 TIMEOUT_RATIO(70%) 경과 시 남은 종목 처리를 중단한다.
        타임아웃으로 스킵된 종목은 None 값으로 결과에 포함된다.

        Args:
            tickers_data: {ticker: yfinance_df} 딕셔너리.
            horizons: 예측 horizon 목록.
            n_samples: Monte Carlo 샘플 수.
            timeout_seconds: 전체 허용 시간(초). 기본 600초.
            fallback_tickers: 우선 처리할 티커 목록.

        Returns:
            {ticker: predict() 반환값 또는 None} 딕셔너리.
            모든 입력 종목이 키로 포함되며, 타임아웃/에러 시 None.
        """
        results: dict[str, Optional[dict[int, dict[str, float]]]] = {}
        deadline = time.monotonic() + timeout_seconds * self.TIMEOUT_RATIO

        # fallback 우선 → 나머지 (tickers_data에 있는 종목만 포함)
        fallback_set = set(fallback_tickers) if fallback_tickers else set()
        fallback = [t for t in tickers_data if t in fallback_set]
        rest = [t for t in tickers_data if t not in fallback_set]
        ordered = fallback + rest

        for ticker in ordered:
            if time.monotonic() > deadline:
                logger.warning(
                    "timeout %.0f%% 경과, 남은 종목 처리 중단. 처리완료: %s",
                    self.TIMEOUT_RATIO * 100,
                    list(results.keys()),
                )
                # 미처리 종목을 모두 None으로 채움
                for remaining in ordered:
                    if remaining not in results:
                        results[remaining] = None
                break

            df = tickers_data.get(ticker)
            if df is None:
                logger.warning("티커 %s: DataFrame 없음, 건너뜀.", ticker)
                results[ticker] = None
                continue

            try:
                result = self.predict(df, horizons=horizons, n_samples=n_samples)
                results[ticker] = result
            except Exception as exc:  # noqa: BLE001
                logger.error("티커 %s 예측 실패: %s", ticker, exc, exc_info=True)
                results[ticker] = None

        return results

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _prepare_input(
        self, df: pd.DataFrame, pred_len: int = 5
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """yfinance DataFrame을 Kronos 입력 형식으로 변환.

        Args:
            df: yfinance 형식 DataFrame (Open/High/Low/Close/Volume, DatetimeIndex).
            pred_len: 예측 기간 (영업일 수).

        Returns:
            (x_df, x_timestamp, y_timestamp)
            - x_df: 소문자 컬럼 DataFrame (open/high/low/close/volume)
            - x_timestamp: 입력 날짜 pd.Series[Timestamp]
            - y_timestamp: 미래 영업일 pd.Series[Timestamp] (pred_len개)
        """
        # tz-aware DatetimeIndex 처리 (yfinance US 종목은 America/New_York tz 포함)
        idx = df.index.tz_localize(None) if hasattr(df.index, "tz") and df.index.tz else df.index

        # 컬럼명 소문자 변환
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        x_df = df.rename(columns=col_map)[["open", "high", "low", "close", "volume"]].copy()

        # x_timestamp: DatetimeIndex → pd.Series (tz 제거된 버전)
        x_ts = pd.Series(idx, name="timestamp")

        # y_timestamp: 마지막 날짜 이후 pred_len개 영업일 생성
        last_date: pd.Timestamp = pd.Timestamp(idx[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
        y_ts = pd.Series(future_dates, name="timestamp")

        return x_df, x_ts, y_ts

    def _compute_statistics(
        self,
        samples: list[list[float]],
        current_price: float,
        horizons: list[int],
    ) -> dict[int, dict[str, float]]:
        """샘플 경로에서 horizon별 통계 계산.

        Args:
            samples: close 가격 경로 목록. 각 경로는 pred_len 길이의 리스트.
            current_price: 현재(마지막 실제) 종가.
            horizons: 통계를 계산할 horizon 목록 (1-indexed 영업일).

        Returns:
            {horizon: {median, p10, p90, direction_prob, volatility}}
        """
        if not samples:
            return {}

        arr = np.array(samples, dtype=np.float64)  # shape: (n_samples, pred_len)
        stats: dict[int, dict[str, float]] = {}

        for h in horizons:
            # horizon은 1-indexed, 배열은 0-indexed
            idx = min(h - 1, arr.shape[1] - 1)
            prices_at_h = arr[:, idx]

            median = float(np.median(prices_at_h))
            p10 = float(np.percentile(prices_at_h, 10))
            p90 = float(np.percentile(prices_at_h, 90))
            direction_prob = float(np.mean(prices_at_h > current_price))
            volatility = float(np.std(prices_at_h) / (current_price + 1e-10))

            stats[h] = {
                "median": median,
                "p10": p10,
                "p90": p90,
                "direction_prob": direction_prob,
                "volatility": volatility,
            }

        return stats
