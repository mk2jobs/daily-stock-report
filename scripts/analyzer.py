"""관심종목 기술적 분석 모듈."""

import json
import sys
import pandas as pd
import numpy as np


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def get_signals(last_row):
    signals = {
        "rsi_signal": "neutral",
        "macd_signal": "neutral",
        "bb_signal": "normal",
        "trend": "sideways",
        "recommendation": "hold"
    }

    rsi = last_row.get('RSI_14')
    if pd.notnull(rsi):
        if rsi > 70:
            signals['rsi_signal'] = 'overbought'
        elif rsi < 30:
            signals['rsi_signal'] = 'oversold'
        elif rsi < 50:
            signals['rsi_signal'] = 'recovering'

    macd = last_row.get('MACD')
    signal = last_row.get('MACD_Signal')
    if pd.notnull(macd) and pd.notnull(signal):
        if macd > signal:
            signals['macd_signal'] = 'bullish'
        elif macd < signal:
            signals['macd_signal'] = 'bearish'

    price = last_row['Close']
    bb_upper = last_row.get('BBU')
    bb_lower = last_row.get('BBL')
    if pd.notnull(bb_upper) and price > bb_upper:
        signals['bb_signal'] = 'overbought'
    elif pd.notnull(bb_lower) and price < bb_lower:
        signals['bb_signal'] = 'oversold'

    sma_50 = last_row.get('SMA_50')
    if pd.notnull(sma_50):
        if price > sma_50 * 1.02:
            signals['trend'] = 'uptrend'
        elif price < sma_50 * 0.98:
            signals['trend'] = 'downtrend'

    score = 0
    if signals['rsi_signal'] == 'oversold':
        score += 2
    elif signals['rsi_signal'] == 'recovering':
        score += 1
    elif signals['rsi_signal'] == 'overbought':
        score -= 2
    if signals['macd_signal'] == 'bullish':
        score += 1
    elif signals['macd_signal'] == 'bearish':
        score -= 1
    if signals['bb_signal'] == 'oversold':
        score += 1
    elif signals['bb_signal'] == 'overbought':
        score -= 1
    if signals['trend'] == 'uptrend':
        score += 1
    elif signals['trend'] == 'downtrend':
        score -= 1

    # 매도 신호는 bearish 지표가 2개 이상일 때만 발동
    bearish_count = sum([
        signals['rsi_signal'] == 'overbought',
        signals['macd_signal'] == 'bearish',
        signals['bb_signal'] == 'overbought',
        signals['trend'] == 'downtrend',
    ])

    if score >= 3:
        signals['recommendation'] = "strong_buy"
    elif score >= 1:
        signals['recommendation'] = "buy"
    elif score <= -3 and bearish_count >= 2:
        signals['recommendation'] = "strong_sell"
    elif score <= -1 and bearish_count >= 2:
        signals['recommendation'] = "sell"

    return signals


def get_long_term_signals(last_row, per, pbr, dividend_yield):
    """장기 투자 관점 신호 (밸류에이션 + 골든/데드크로스)."""
    signals = {
        "cross": "none",
        "valuation": "neutral",
        "recommendation": "hold",
    }

    sma_50 = last_row.get('SMA_50')
    sma_200 = last_row.get('SMA_200')
    if pd.notnull(sma_50) and pd.notnull(sma_200):
        if sma_50 > sma_200:
            signals['cross'] = 'golden_cross'
        else:
            signals['cross'] = 'dead_cross'

    score = 0

    # 골든/데드크로스
    if signals['cross'] == 'golden_cross':
        score += 2
    elif signals['cross'] == 'dead_cross':
        score -= 2

    # PER 밸류에이션
    if per is not None:
        if per < 10:
            signals['valuation'] = 'undervalued'
            score += 1
        elif per > 30:
            signals['valuation'] = 'overvalued'
            score -= 1

    # PBR
    if pbr is not None:
        if pbr < 1:
            score += 1
        elif pbr > 5:
            score -= 1

    # 배당수익률 (원본은 0~1 비율, 여기선 이미 % 변환 전 원본)
    if dividend_yield is not None:
        if dividend_yield > 0.03:
            score += 1
        elif dividend_yield > 0.02:
            score += 0.5

    if score >= 3:
        signals['recommendation'] = 'strong_buy'
    elif score >= 1:
        signals['recommendation'] = 'buy'
    elif score <= -3:
        signals['recommendation'] = 'strong_sell'
    elif score <= -1:
        signals['recommendation'] = 'sell'

    return signals


def analyze_stock(ticker, period="1y"):
    """단일 종목 기술적 분석. 실패 시 None 반환."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval="1d", auto_adjust=True)

        if hist.empty or len(hist) < 20:
            return None

        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['RSI_14'] = calculate_rsi(hist['Close'])
        hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = calculate_macd(hist['Close'])
        hist['BBU'], hist['BBM'], hist['BBL'] = calculate_bollinger_bands(hist['Close'])

        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        signals = get_signals(last)

        change = last['Close'] - prev['Close']
        change_pct = (change / prev['Close']) * 100 if prev['Close'] != 0 else 0

        info = stock.info or {}
        currency = info.get('currency', 'KRW' if '.K' in ticker else 'USD')

        # 장기 지표
        per = info.get('trailingPE') or info.get('forwardPE')
        pbr = info.get('priceToBook')
        dividend_yield = info.get('dividendYield')
        # yfinance 한국 주식 비정상 값 필터링
        if dividend_yield and dividend_yield > 0.20:
            dividend_yield = None
        if per and (per < 0 or per > 1000):
            per = None

        long_term = get_long_term_signals(last, per, pbr, dividend_yield)

        return {
            "ticker": ticker,
            "currency": currency,
            "price": round(float(last['Close']), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "volume": int(last['Volume']),
            "rsi_14": round(float(last['RSI_14']), 2) if pd.notnull(last.get('RSI_14')) else None,
            "macd": round(float(last['MACD']), 3) if pd.notnull(last.get('MACD')) else None,
            "macd_signal": round(float(last['MACD_Signal']), 3) if pd.notnull(last.get('MACD_Signal')) else None,
            "macd_hist": round(float(last['MACD_Hist']), 3) if pd.notnull(last.get('MACD_Hist')) else None,
            "sma_20": round(float(last['SMA_20']), 2) if pd.notnull(last.get('SMA_20')) else None,
            "sma_50": round(float(last['SMA_50']), 2) if pd.notnull(last.get('SMA_50')) else None,
            "sma_200": round(float(last['SMA_200']), 2) if pd.notnull(last.get('SMA_200')) else None,
            "bb_upper": round(float(last['BBU']), 2) if pd.notnull(last.get('BBU')) else None,
            "bb_lower": round(float(last['BBL']), 2) if pd.notnull(last.get('BBL')) else None,
            "signals": signals,
            "per": round(float(per), 2) if per else None,
            "pbr": round(float(pbr), 2) if pbr else None,
            "dividend_yield": round(float(dividend_yield) * 100, 2) if dividend_yield else None,
            "long_term": long_term,
        }
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}", file=sys.stderr)
        return None


def analyze_watchlist(watchlist_path):
    """watchlist.json 기반 전체 분석."""
    with open(watchlist_path, 'r') as f:
        watchlist = json.load(f)

    results = []
    for stock in watchlist['stocks']:
        print(f"Analyzing {stock['name']} ({stock['ticker']})...", file=sys.stderr)
        data = analyze_stock(stock['ticker'])
        if data:
            data['name'] = stock['name']
            data['market'] = stock['market']
            results.append(data)

    return results
