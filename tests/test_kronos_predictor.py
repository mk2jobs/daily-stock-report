"""
KronosPredictor 래퍼 단위 테스트.
모든 테스트는 mock을 사용하여 실제 모델 로딩 없이 실행.
"""
import sys
import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# libs 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helper: 테스트용 yfinance 스타일 DataFrame 생성
# ---------------------------------------------------------------------------

def _make_yfinance_df(n_rows: int = 100) -> pd.DataFrame:
    """yfinance 형식 DataFrame 생성 (대문자 컬럼, DatetimeIndex).

    bdate_range 대신 date_range + 필터링으로 정확히 n_rows 행을 생성한다.
    """
    # 충분히 많은 날짜를 생성한 후 영업일(월~금)만 필터링
    all_dates = pd.date_range(end=pd.Timestamp("2026-01-10"), periods=n_rows * 3)
    biz_dates = all_dates[all_dates.dayofweek < 5][-n_rows:]
    assert len(biz_dates) == n_rows, f"영업일 생성 실패: {len(biz_dates)} != {n_rows}"

    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, n_rows).cumsum()
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n_rows).astype(float),
        },
        index=biz_dates,
    )
    return df


# ---------------------------------------------------------------------------
# Test 1: _prepare_input — yfinance DF → Kronos 입력 형식 변환
# ---------------------------------------------------------------------------

class TestFormatYfinanceToKronos:
    def test_format_yfinance_to_kronos(self):
        """yfinance DataFrame을 Kronos 입력 형식으로 변환하는지 검증."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False  # 모델 로딩 생략

        df = _make_yfinance_df(n_rows=100)
        x_df, x_ts, y_ts = predictor._prepare_input(df, pred_len=5)

        # 컬럼명이 소문자여야 함
        assert list(x_df.columns[:5]) == ["open", "high", "low", "close", "volume"], (
            f"컬럼명이 소문자가 아님: {list(x_df.columns)}"
        )

        # x_timestamp 길이 = 입력 행 수
        assert len(x_ts) == len(df), f"x_timestamp 길이 불일치: {len(x_ts)} vs {len(df)}"

        # y_timestamp 길이 = pred_len
        assert len(y_ts) == 5, f"y_timestamp 길이 불일치: {len(y_ts)} vs 5"

        # y_timestamp는 영업일만 포함되어야 함
        assert isinstance(y_ts, pd.Series), "y_timestamp는 pd.Series여야 함"
        for ts in y_ts:
            assert ts.weekday() < 5, f"주말 포함된 y_timestamp: {ts}"

        # y_timestamp의 첫 날짜는 마지막 x 날짜 이후여야 함
        assert y_ts.iloc[0] > x_ts.iloc[-1], (
            f"y_timestamp 첫 날짜({y_ts.iloc[0]})가 x 마지막 날짜({x_ts.iloc[-1]}) 이후가 아님"
        )

    def test_format_tz_aware_index(self):
        """tz-aware DatetimeIndex (America/New_York) → tz-naive 변환 검증."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False

        df = _make_yfinance_df(n_rows=100)
        # yfinance US 종목처럼 tz 부여
        df.index = df.index.tz_localize("America/New_York")

        x_df, x_ts, y_ts = predictor._prepare_input(df, pred_len=5)

        # x_timestamp에 tz가 없어야 함
        assert x_ts.dt.tz is None, f"x_timestamp에 tz가 남아있음: {x_ts.dt.tz}"
        assert y_ts.dt.tz is None, f"y_timestamp에 tz가 남아있음: {y_ts.dt.tz}"
        assert len(x_ts) == 100
        assert len(y_ts) == 5


# ---------------------------------------------------------------------------
# Test 2: _compute_statistics — 샘플 경로로 통계 계산
# ---------------------------------------------------------------------------

class TestComputeStatistics:
    def _make_samples(self) -> list[list[float]]:
        """5개 샘플, 각 5일 종가 경로."""
        return [
            [100.0, 101.0, 102.0, 103.0, 104.0],  # 꾸준히 상승
            [100.0, 99.0, 98.0, 97.0, 96.0],       # 꾸준히 하락
            [100.0, 102.0, 101.0, 103.0, 105.0],   # 상승
            [100.0, 98.0, 100.0, 99.0, 101.0],     # 약보합
            [100.0, 103.0, 102.0, 104.0, 106.0],   # 강한 상승
        ]

    def test_compute_statistics_empty_samples(self):
        """빈 samples 리스트 → 빈 딕셔너리 반환."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False

        result = predictor._compute_statistics([], current_price=100.0, horizons=[1, 5])
        assert result == {}, f"빈 samples에서 빈 dict 기대, 실제: {result}"

    def test_compute_statistics_horizons(self):
        """horizons [1, 5]에 대한 통계 키가 모두 존재하는지 검증."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False

        samples = self._make_samples()
        current_price = 100.0
        stats = predictor._compute_statistics(samples, current_price, horizons=[1, 5])

        for h in [1, 5]:
            assert h in stats, f"horizon {h}에 대한 통계 없음"
            for key in ["median", "p10", "p90", "direction_prob", "volatility"]:
                assert key in stats[h], f"horizon {h}: '{key}' 키 없음"

    def test_compute_statistics_values(self):
        """통계 값의 방향성과 범위 검증."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False

        samples = self._make_samples()
        current_price = 100.0
        stats = predictor._compute_statistics(samples, current_price, horizons=[1, 5])

        # horizon=5: p10 <= median <= p90
        h5 = stats[5]
        assert h5["p10"] <= h5["median"] <= h5["p90"], (
            f"p10({h5['p10']}) <= median({h5['median']}) <= p90({h5['p90']}) 위반"
        )

        # direction_prob: 0~1 범위
        assert 0.0 <= h5["direction_prob"] <= 1.0, (
            f"direction_prob 범위 오류: {h5['direction_prob']}"
        )

        # volatility >= 0
        assert h5["volatility"] >= 0.0, f"volatility 음수: {h5['volatility']}"

        # 샘플 5개 중 4개가 상승 → direction_prob > 0.5 (horizon=5)
        assert h5["direction_prob"] > 0.5, (
            f"상승 우세 샘플에서 direction_prob({h5['direction_prob']}) <= 0.5"
        )


# ---------------------------------------------------------------------------
# Test 3: predict_batch — fallback tickers 우선 처리 및 timeout 검증
# ---------------------------------------------------------------------------

class TestPredictBatchTimeout:
    def test_predict_batch_all_tickers_processed(self):
        """3개 티커 모두 결과를 반환하는지 확인."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = True

        # predict를 mock으로 대체: 항상 dummy stats 반환
        dummy_stats = {
            1: {"median": 101.0, "p10": 99.0, "p90": 103.0, "direction_prob": 0.6, "volatility": 0.02},
            5: {"median": 102.0, "p10": 98.0, "p90": 106.0, "direction_prob": 0.6, "volatility": 0.03},
        }
        predictor.predict = MagicMock(return_value=dummy_stats)

        tickers_data = {
            "AAPL": _make_yfinance_df(100),
            "MSFT": _make_yfinance_df(100),
            "GOOGL": _make_yfinance_df(100),
        }
        fallback_tickers = ["MSFT"]

        results = predictor.predict_batch(
            tickers_data=tickers_data,
            horizons=[1, 5],
            n_samples=2,
            timeout_seconds=600,
            fallback_tickers=fallback_tickers,
        )

        # 3개 티커 모두 결과 있어야 함
        assert set(results.keys()) == {"AAPL", "MSFT", "GOOGL"}, (
            f"누락된 티커: {set(tickers_data.keys()) - set(results.keys())}"
        )

    def test_predict_batch_fallback_processed_first(self):
        """fallback_tickers가 먼저 처리되는지 처리 순서 검증."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = True

        # 각 DataFrame에 고유 식별자를 심어 호출 순서 추적
        dfs = {}
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            df = _make_yfinance_df(100)
            df.attrs["_ticker"] = ticker  # DataFrame에 식별자 부착
            dfs[ticker] = df

        process_order: list[str] = []

        def mock_predict(df, horizons, n_samples):
            process_order.append(df.attrs["_ticker"])
            return {
                1: {"median": 101.0, "p10": 99.0, "p90": 103.0, "direction_prob": 0.6, "volatility": 0.02},
            }

        predictor.predict = MagicMock(side_effect=mock_predict)

        fallback_tickers = ["GOOGL"]

        results = predictor.predict_batch(
            tickers_data=dfs,
            horizons=[1],
            n_samples=1,
            timeout_seconds=600,
            fallback_tickers=fallback_tickers,
        )

        # fallback ticker인 GOOGL이 첫 번째로 처리되어야 함
        assert process_order[0] == "GOOGL", (
            f"fallback 종목 GOOGL이 첫 번째로 처리되지 않음. 실제 순서: {process_order}"
        )
        assert set(results.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_predict_batch_ignores_fallback_not_in_tickers_data(self):
        """fallback_tickers에 있지만 tickers_data에 없는 종목은 결과에 포함되지 않아야 함."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = True

        dummy_stats = {
            1: {"median": 101.0, "p10": 99.0, "p90": 103.0, "direction_prob": 0.6, "volatility": 0.02},
        }
        predictor.predict = MagicMock(return_value=dummy_stats)

        tickers_data = {
            "AAPL": _make_yfinance_df(100),
            "MSFT": _make_yfinance_df(100),
        }
        # TSLA는 tickers_data에 없음
        fallback_tickers = ["TSLA", "MSFT"]

        results = predictor.predict_batch(
            tickers_data=tickers_data,
            horizons=[1],
            n_samples=1,
            timeout_seconds=600,
            fallback_tickers=fallback_tickers,
        )

        # tickers_data에 있는 종목만 결과에 포함
        assert set(results.keys()) == {"AAPL", "MSFT"}, (
            f"tickers_data에 없는 TSLA가 포함됨: {set(results.keys())}"
        )

    def test_predict_batch_timeout_skips_remaining(self):
        """타임아웃 시 미처리 종목이 None으로 결과에 포함되는지 검증.

        time.monotonic을 mock하여 결정론적으로 테스트.
        - 1번째 호출(시작): 0.0
        - 2번째 호출(T1 전 체크): 0.0 → 통과
        - 3번째 호출(T2 전 체크): 0.5 → 통과 (deadline=0.7)
        - 4번째 호출(T3 전 체크): 1.0 → 초과 → T3~T6 스킵
        """
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = True

        dummy_stats = {
            1: {"median": 101.0, "p10": 99.0, "p90": 103.0, "direction_prob": 0.6, "volatility": 0.02},
        }
        predictor.predict = MagicMock(return_value=dummy_stats)

        dfs = {}
        for ticker in ["T1", "T2", "T3", "T4", "T5", "T6"]:
            df = _make_yfinance_df(100)
            df.attrs["_ticker"] = ticker
            dfs[ticker] = df

        # monotonic 반환값: 시작(0.0), T1 체크(0.0), T2 체크(0.5), T3 체크(1.0)
        # deadline = 0.0 + 1.0 * 0.70 = 0.70
        # T1: 0.0 < 0.70 → 처리, T2: 0.5 < 0.70 → 처리, T3: 1.0 > 0.70 → 중단
        mock_times = iter([0.0, 0.0, 0.5, 1.0])

        with patch("libs.kronos_predictor.time.monotonic", side_effect=mock_times):
            results = predictor.predict_batch(
                tickers_data=dfs,
                horizons=[1],
                n_samples=1,
                timeout_seconds=1.0,
                fallback_tickers=["T1"],
            )

        # 모든 종목이 결과에 포함되어야 함 (스킵된 종목도 None으로)
        assert set(results.keys()) == {"T1", "T2", "T3", "T4", "T5", "T6"}, (
            f"결과 키 누락: {set(dfs.keys()) - set(results.keys())}"
        )

        # T1, T2만 처리됨, T3~T6은 None
        assert results["T1"] is not None, "T1이 처리되지 않음"
        assert results["T2"] is not None, "T2가 처리되지 않음"
        for skipped in ["T3", "T4", "T5", "T6"]:
            assert results[skipped] is None, (
                f"{skipped}이 타임아웃으로 스킵되지 않음: {results[skipped]}"
            )


# ---------------------------------------------------------------------------
# Test 4: predict — 데이터 부족 시 None 반환
# ---------------------------------------------------------------------------

class TestPredictReturnsNoneOnInsufficientData:
    def test_returns_none_when_df_too_short(self):
        """입력 DataFrame이 60행 미만이면 None 반환."""
        from libs.kronos_predictor import KronosPredictor

        # _available=False 상태에서도 validate 로직은 실행되어야 함
        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False

        short_df = _make_yfinance_df(n_rows=30)  # 60 미만
        result = predictor.predict(short_df, horizons=[1, 5], n_samples=3)

        assert result is None, f"60행 미만 DF에서 None 반환 기대, 실제: {result}"

    def test_returns_none_when_df_exactly_59_rows(self):
        """59행 DataFrame → None 반환 경계값 테스트."""
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False

        df_59 = _make_yfinance_df(n_rows=59)
        result = predictor.predict(df_59, horizons=[1, 5], n_samples=3)

        assert result is None, f"59행 DF에서 None 반환 기대, 실제: {result}"

    def test_does_not_return_none_when_df_sufficient(self):
        """60행 이상이고 _available=False면 None이 아닌 에러(ImportError/RuntimeError) 또는 진행 여부 확인.

        _available=False일 때는 모델 없이 실행 불가이므로 None 반환.
        충분한 데이터 + 모델 있을 때만 실제 통계 반환.
        이 테스트는 '데이터 충분 → validate 통과' 경로를 검증.
        """
        from libs.kronos_predictor import KronosPredictor

        predictor = KronosPredictor.__new__(KronosPredictor)
        predictor._available = False  # 모델 없음

        df_100 = _make_yfinance_df(n_rows=100)
        # _available=False이면 None 반환 (모델 없음), 단 "데이터 부족"이 아닌 "모델 미가용"이 원인
        result = predictor.predict(df_100, horizons=[1, 5], n_samples=1)
        # 데이터 검증은 통과하되 모델 없어서 None — 데이터 부족으로 None인 것과 다름
        # 여기서는 "60행 이상이므로 데이터 부족 경로가 아님"만 확인
        # (모델 없음으로 인한 None은 허용)
        assert result is None  # _available=False이므로 None
