"""시장 요약: 주요 지수, 환율, VIX, 섹터 흐름."""

import sys
from datetime import datetime, timezone, timedelta
import pandas as pd


def get_market_indices():
    """코스피, 코스닥, S&P500, 나스닥 지수."""
    import yfinance as yf

    indices = [
        {"name": "코스피", "ticker": "^KS11"},
        {"name": "코스닥", "ticker": "^KQ11"},
        {"name": "S&P 500", "ticker": "^GSPC"},
        {"name": "나스닥", "ticker": "^IXIC"},
        {"name": "다우존스", "ticker": "^DJI"},
    ]

    results = []
    for idx in indices:
        try:
            t = yf.Ticker(idx['ticker'])
            hist = t.history(period="5d", interval="1d")
            if len(hist) >= 2:
                last = hist.iloc[-1]['Close']
                prev = hist.iloc[-2]['Close']
                change_pct = ((last - prev) / prev) * 100
                results.append({
                    "name": idx['name'],
                    "value": round(float(last), 2),
                    "change_pct": round(float(change_pct), 2),
                })
        except Exception as e:
            print(f"[INDEX] {idx['name']} skip: {e}", file=sys.stderr)

    return results


def get_exchange_rates():
    """USD/KRW, JPY/KRW 환율."""
    import yfinance as yf

    rates = [
        {"name": "USD/KRW", "ticker": "KRW=X"},
        {"name": "JPY/KRW", "ticker": "JPYKRW=X"},
    ]

    results = []
    for r in rates:
        try:
            t = yf.Ticker(r['ticker'])
            hist = t.history(period="5d", interval="1d")
            if len(hist) >= 2:
                last = hist.iloc[-1]['Close']
                prev = hist.iloc[-2]['Close']
                change_pct = ((last - prev) / prev) * 100
                results.append({
                    "name": r['name'],
                    "value": round(float(last), 2),
                    "change_pct": round(float(change_pct), 2),
                })
        except Exception as e:
            print(f"[FX] {r['name']} skip: {e}", file=sys.stderr)

    return results


def get_vix():
    """VIX (변동성/공포 지수)."""
    import yfinance as yf

    try:
        t = yf.Ticker("^VIX")
        hist = t.history(period="5d", interval="1d")
        if len(hist) >= 2:
            last = hist.iloc[-1]['Close']
            prev = hist.iloc[-2]['Close']
            change_pct = ((last - prev) / prev) * 100

            if last >= 30:
                level = "극도의 공포"
            elif last >= 20:
                level = "공포/불안"
            elif last >= 15:
                level = "보통"
            else:
                level = "안일/낙관"

            return {
                "value": round(float(last), 2),
                "change_pct": round(float(change_pct), 2),
                "level": level,
            }
    except Exception as e:
        print(f"[VIX] skip: {e}", file=sys.stderr)

    return None


def get_sector_performance():
    """주요 섹터 ETF 기반 흐름."""
    import yfinance as yf

    sectors = [
        {"name": "IT/기술", "ticker": "XLK"},
        {"name": "반도체", "ticker": "SOXX"},
        {"name": "헬스케어", "ticker": "XLV"},
        {"name": "금융", "ticker": "XLF"},
        {"name": "소비재", "ticker": "XLY"},
        {"name": "에너지", "ticker": "XLE"},
        {"name": "산업재", "ticker": "XLI"},
        {"name": "커뮤니케이션", "ticker": "XLC"},
    ]

    results = []
    for s in sectors:
        try:
            t = yf.Ticker(s['ticker'])
            hist = t.history(period="5d", interval="1d")
            if len(hist) >= 2:
                last = hist.iloc[-1]['Close']
                prev = hist.iloc[-2]['Close']
                change_pct = ((last - prev) / prev) * 100
                results.append({
                    "name": s['name'],
                    "ticker": s['ticker'],
                    "change_pct": round(float(change_pct), 2),
                })
        except Exception as e:
            print(f"[SECTOR] {s['name']} skip: {e}", file=sys.stderr)

    results.sort(key=lambda x: x['change_pct'], reverse=True)
    return results
