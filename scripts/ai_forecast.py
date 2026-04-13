"""
AI 예측 섹션 생성기 — Kronos 모델을 이용한 배치 예측 결과를 HTML로 변환.

report_generator.py에서 호출:
    from ai_forecast import generate_ai_forecast_section
    html = generate_ai_forecast_section(results, ticker_info_cache, timeout_seconds=600)
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional

logger = logging.getLogger(__name__)

# FALLBACK_TICKERS: predict_batch에서 우선 처리할 종목 (KRX 10 → US 5 순서)
FALLBACK_TICKERS: list[str] = [
    "005930.KS", "000660.KS", "005380.KS", "051910.KS", "373220.KS",
    "006400.KS", "035420.KS", "035720.KS", "017670.KS", "259960.KS",
    "NVDA", "AAPL", "MSFT", "GOOGL", "TSLA",
]


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------

def _format_price(price: float, currency: str) -> str:
    """통화에 맞게 가격을 포맷.

    Args:
        price: 가격 수치
        currency: 통화 코드 ("KRW" 또는 기타)

    Returns:
        포맷된 가격 문자열. KRW: "₩58,500" / USD: "$138.25"
    """
    if currency == "KRW":
        return f"₩{price:,.0f}"
    return f"${price:,.2f}"


def _format_range(p10: float, p90: float, currency: str) -> str:
    """예측 범위(p10~p90)를 포맷.

    Args:
        p10: 10번째 백분위 가격
        p90: 90번째 백분위 가격
        currency: 통화 코드

    Returns:
        포맷된 범위 문자열. KRW: "58,100~60,300" / USD: "$135.0~$142.0"
    """
    if currency == "KRW":
        return f"{p10:,.0f}~{p90:,.0f}"
    return f"${p10:,.1f}~${p90:,.1f}"


def _direction_icon(prob: float) -> str:
    """상승 확률에 따른 방향 아이콘 반환.

    Args:
        prob: 상승 확률 (0.0~1.0)

    Returns:
        "🔼" (≥0.55), "🔽" (≤0.45), "➡️" (그 외)
    """
    if prob >= 0.55:
        return "🔼"
    elif prob <= 0.45:
        return "🔽"
    return "➡️"


def _volatility_label(vol: float) -> str:
    """변동성 수치에 따른 라벨 반환.

    Args:
        vol: 변동성 수치 (std/price 비율)

    Returns:
        "높음" (>0.03), "보통" (0.015~0.03), "낮음" (<0.015)
    """
    if vol > 0.03:
        return "높음"
    elif vol >= 0.015:
        return "보통"
    return "낮음"


# ---------------------------------------------------------------------------
# HTML 빌더
# ---------------------------------------------------------------------------

def build_forecast_html(
    forecasts: dict[str, Optional[dict[int, dict[str, float]]]],
    stocks: list[dict],
) -> str:
    """예측 결과를 HTML 섹션으로 변환.

    Args:
        forecasts: {ticker: predict() 반환값 또는 None}
            predict() 반환값: {1: {median, p10, p90, direction_prob, volatility},
                               5: {median, p10, p90, direction_prob, volatility}}
        stocks: 정렬된 관심종목 리스트. 각 항목: {ticker, name, market, price, currency, ...}

    Returns:
        HTML 문자열. forecasts가 비어있으면 "".
    """
    if not forecasts:
        return ""

    # stocks 중 forecasts에 키가 있는 종목만 표시 대상으로 추림
    display_stocks = [s for s in stocks if s["ticker"] in forecasts]
    if not display_stocks:
        return ""

    skip_count = sum(1 for s in display_stocks if forecasts.get(s["ticker"]) is None)

    # 공통 테이블 스타일 (이메일 클라이언트 호환 인라인 CSS)
    table_style = (
        "width:100%;border-collapse:collapse;font-size:13px;"
        "font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;"
    )
    th_style = (
        "background:#2C2620;color:#E8E2DB;padding:9px 8px;text-align:left;"
        "font-size:11px;font-weight:500;letter-spacing:0.5px;"
    )
    td_style = (
        "padding:9px 8px;border-bottom:1px solid #E8E2DB;"
        "color:#3D3529;font-size:13px;"
    )
    skip_style = "color:#9B8E7E;font-size:12px;"

    # ------------------------------------------------------------------ 1일 예측
    table_1d = f'<table class="ai-table" style="{table_style}">'
    table_1d += (
        f'<tr>'
        f'<th style="{th_style}border-radius:6px 0 0 0">종목</th>'
        f'<th style="{th_style}">현재가</th>'
        f'<th style="{th_style}">예상 범위</th>'
        f'<th style="{th_style}">방향</th>'
        f'<th style="{th_style}border-radius:0 6px 0 0">상승확률</th>'
        f'</tr>'
    )

    for i, s in enumerate(display_stocks):
        ticker = s["ticker"]
        currency = s.get("currency", "USD")
        current_price = s.get("price", 0)
        row_bg = "background:#FDFBF8;" if i % 2 == 0 else ""
        result = forecasts.get(ticker)

        if result is None or 1 not in result:
            table_1d += (
                f'<tr style="{row_bg}">'
                f'<td style="{td_style}font-weight:600">{s["name"]}<br>'
                f'<span style="color:#9B8E7E;font-size:10px">{ticker}</span></td>'
                f'<td style="{td_style}">{_format_price(current_price, currency)}</td>'
                f'<td style="{td_style}{skip_style}" colspan="3">⏱ 스킵</td>'
                f'</tr>'
            )
        else:
            h1 = result[1]
            direction = _direction_icon(h1["direction_prob"])
            prob_pct = f'{h1["direction_prob"] * 100:.0f}%'
            price_range = _format_range(h1["p10"], h1["p90"], currency)
            table_1d += (
                f'<tr style="{row_bg}">'
                f'<td style="{td_style}font-weight:600">{s["name"]}<br>'
                f'<span style="color:#9B8E7E;font-size:10px">{ticker}</span></td>'
                f'<td style="{td_style}">{_format_price(current_price, currency)}</td>'
                f'<td style="{td_style}">{price_range}</td>'
                f'<td style="{td_style}font-size:16px">{direction}</td>'
                f'<td style="{td_style}">{prob_pct}</td>'
                f'</tr>'
            )

    table_1d += "</table>"

    # ------------------------------------------------------------------ 5일 예측
    table_5d = f'<table class="ai-table" style="{table_style}margin-top:16px">'
    table_5d += (
        f'<tr>'
        f'<th style="{th_style}border-radius:6px 0 0 0">종목</th>'
        f'<th style="{th_style}">현재가</th>'
        f'<th style="{th_style}">예상 범위</th>'
        f'<th style="{th_style}">방향</th>'
        f'<th style="{th_style}border-radius:0 6px 0 0">변동성</th>'
        f'</tr>'
    )

    for i, s in enumerate(display_stocks):
        ticker = s["ticker"]
        currency = s.get("currency", "USD")
        current_price = s.get("price", 0)
        row_bg = "background:#FDFBF8;" if i % 2 == 0 else ""
        result = forecasts.get(ticker)

        if result is None or 5 not in result:
            table_5d += (
                f'<tr style="{row_bg}">'
                f'<td style="{td_style}font-weight:600">{s["name"]}<br>'
                f'<span style="color:#9B8E7E;font-size:10px">{ticker}</span></td>'
                f'<td style="{td_style}">{_format_price(current_price, currency)}</td>'
                f'<td style="{td_style}{skip_style}" colspan="3">⏱ 스킵</td>'
                f'</tr>'
            )
        else:
            h5 = result[5]
            direction = _direction_icon(h5["direction_prob"])
            vol_label = _volatility_label(h5["volatility"])
            price_range = _format_range(h5["p10"], h5["p90"], currency)
            table_5d += (
                f'<tr style="{row_bg}">'
                f'<td style="{td_style}font-weight:600">{s["name"]}<br>'
                f'<span style="color:#9B8E7E;font-size:10px">{ticker}</span></td>'
                f'<td style="{td_style}">{_format_price(current_price, currency)}</td>'
                f'<td style="{td_style}">{price_range}</td>'
                f'<td style="{td_style}font-size:16px">{direction}</td>'
                f'<td style="{td_style}">{vol_label}</td>'
                f'</tr>'
            )

    table_5d += "</table>"

    # ------------------------------------------------------------------ 푸터
    footer_note = "⚠ AI 예측은 참고용이며 투자 권유가 아닙니다."
    if skip_count > 0:
        footer_note += f" ({skip_count}개 종목 타임아웃으로 스킵됨)"

    footer_style = (
        "font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;"
        "font-size:11px;color:#9B8E7E;margin-top:12px;"
    )

    # ------------------------------------------------------------------ 섹션 조립
    section_style = (
        "border:1px solid #E8E2DB;border-radius:10px;"
        "padding:20px 24px;margin:0;"
    )
    sub_title_style = (
        "font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;"
        "font-size:11px;letter-spacing:1px;text-transform:uppercase;"
        "color:#9B8E7E;margin:0 0 10px 0;font-weight:600;"
    )

    html = f'<div style="{section_style}">'
    html += f'<p style="{sub_title_style}">1일 예측</p>'
    html += table_1d
    html += f'<p style="{sub_title_style}margin-top:20px;">5일(주간) 예측</p>'
    html += table_5d
    html += f'<p style="{footer_style}">{footer_note}</p>'
    html += "</div>"

    return html


# ---------------------------------------------------------------------------
# 메인 공개 함수
# ---------------------------------------------------------------------------

def generate_ai_forecast_section(
    sorted_watchlist: list[dict],
    ticker_info_cache: Optional[dict] = None,
    timeout_seconds: int = 600,
) -> str:
    """Kronos 예측 결과를 HTML [AI 예측] 섹션으로 생성.

    Args:
        sorted_watchlist: analyze_watchlist() + sort_by_market_and_cap() 결과.
            각 항목: {ticker, name, market, price, currency, ...}
        ticker_info_cache: fetch_market_caps()가 반환한 캐시 (사용하지 않지만 시그니처 일치용).
        timeout_seconds: predict_batch() 전체 허용 시간(초). 기본 600초.

    Returns:
        HTML 문자열. Kronos 모델 미가용 또는 결과 없으면 "".
    """
    # libs 경로 추가 (scripts/ 기준으로 한 단계 위 libs/)
    _libs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "libs")
    if _libs_path not in sys.path:
        sys.path.insert(0, _libs_path)

    try:
        from kronos_predictor import KronosPredictor
    except ImportError as exc:
        logger.warning("KronosPredictor import 실패, AI 예측 섹션 생략: %s", exc)
        return ""

    predictor = KronosPredictor()
    if not predictor._available:
        logger.warning("Kronos 모델 미가용, AI 예측 섹션 생략.")
        return ""

    # yfinance로 90일 OHLCV 데이터 수집
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance import 실패, AI 예측 섹션 생략.")
        return ""

    # FALLBACK_TICKERS에 해당하는 종목만 예측 대상으로 제한
    fallback_set = set(FALLBACK_TICKERS)
    target_stocks = [s for s in sorted_watchlist if s.get("ticker") in fallback_set]

    tickers_data: dict = {}
    for stock in target_stocks:
        ticker = stock.get("ticker")
        if not ticker:
            continue
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="90d", auto_adjust=True)
            if df is not None and len(df) >= 60:
                tickers_data[ticker] = df
            else:
                logger.debug("티커 %s: 데이터 부족 (%d행), 건너뜀.", ticker, len(df) if df is not None else 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("티커 %s yfinance 다운로드 실패: %s", ticker, exc)

    if not tickers_data:
        logger.warning("수집된 OHLCV 데이터 없음, AI 예측 섹션 생략.")
        return ""

    # 배치 예측 실행
    logger.info("Kronos 배치 예측 시작: %d종목, timeout=%ds", len(tickers_data), timeout_seconds)
    forecasts = predictor.predict_batch(
        tickers_data=tickers_data,
        horizons=[1, 5],
        n_samples=5,
        timeout_seconds=float(timeout_seconds),
        fallback_tickers=set(FALLBACK_TICKERS),
    )

    return build_forecast_html(forecasts, target_stocks)
