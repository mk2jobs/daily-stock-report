"""숨겨진 종목 추천 스캐너. 거래량 급증 + RSI 과매도 종목 탐색."""

import sys
import pandas as pd
import numpy as np


# KOSPI/KOSDAQ 주요 종목 풀 (시가총액 상위 + 중소형)
SCAN_POOL = [
    # KOSPI 대형
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
    "000270.KS", "068270.KS", "105560.KS", "005490.KS", "035420.KS",
    "006400.KS", "051910.KS", "035720.KS", "012330.KS", "028260.KS",
    "015760.KS", "055550.KS", "086790.KS", "066570.KS", "017670.KS",
    "009150.KS", "012450.KS", "034730.KS", "003550.KS", "032830.KS",
    "138040.KS", "003670.KS", "010130.KS", "011200.KS", "036570.KS",
    # KOSPI 중형
    "033780.KS", "010950.KS", "034020.KS", "004020.KS", "016360.KS",
    "009540.KS", "000810.KS", "097950.KS", "010140.KS", "002790.KS",
    # KOSDAQ
    "247540.KQ", "086520.KQ", "066970.KQ", "028300.KQ", "196170.KQ",
    "058470.KQ", "035900.KQ", "293490.KQ", "263750.KQ", "068760.KQ",
    "042700.KQ", "112040.KQ", "039030.KQ", "095340.KQ", "036930.KQ",
    "041510.KQ", "005290.KQ", "222080.KQ", "078600.KQ", "048410.KQ",
]

# 종목명 매핑
TICKER_NAMES = {
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "373220.KS": "LG에너지솔루션",
    "207940.KS": "삼성바이오로직스", "005380.KS": "현대차", "000270.KS": "기아",
    "068270.KS": "셀트리온", "105560.KS": "KB금융", "005490.KS": "POSCO홀딩스",
    "035420.KS": "네이버", "006400.KS": "삼성SDI", "051910.KS": "LG화학",
    "035720.KS": "카카오", "012330.KS": "현대모비스", "028260.KS": "삼성물산",
    "015760.KS": "한국전력", "055550.KS": "신한지주", "086790.KS": "하나금융지주",
    "066570.KS": "LG전자", "017670.KS": "SK텔레콤", "009150.KS": "삼성전기",
    "012450.KS": "한화에어로스페이스", "034730.KS": "SK", "003550.KS": "LG",
    "032830.KS": "삼성생명", "138040.KS": "메리츠금융지주", "003670.KS": "포스코퓨처엠",
    "010130.KS": "고려아연", "011200.KS": "HMM", "036570.KS": "엔씨소프트",
    "033780.KS": "KT&G", "010950.KS": "S-Oil", "034020.KS": "두산에너빌리티",
    "004020.KS": "현대제철", "016360.KS": "삼성증권", "009540.KS": "HD한국조선해양",
    "000810.KS": "삼성화재", "097950.KS": "CJ제일제당", "010140.KS": "삼성중공업",
    "002790.KS": "아모레G",
    "247540.KQ": "에코프로비엠", "086520.KQ": "에코프로", "066970.KQ": "엘앤에프",
    "028300.KQ": "HLB", "196170.KQ": "알테오젠", "058470.KQ": "리노공업",
    "035900.KQ": "JYP Ent.", "293490.KQ": "카카오게임즈", "263750.KQ": "펄어비스",
    "068760.KQ": "셀트리온제약", "042700.KQ": "한미반도체", "112040.KQ": "위메이드",
    "039030.KQ": "이오테크닉스", "095340.KQ": "ISC", "036930.KQ": "주성엔지니어링",
    "041510.KQ": "에스엠", "005290.KQ": "동진쎄미켐", "222080.KQ": "씨아이에스",
    "078600.KQ": "대주전자재료", "048410.KQ": "현대바이오",
}


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def scan_gems(top_n=10):
    """거래량 급증 + RSI 과매도 종목 스캔."""
    import yfinance as yf

    gems = []

    for ticker in SCAN_POOL:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo", interval="1d", auto_adjust=True)

            if hist.empty or len(hist) < 30:
                continue

            hist['RSI_14'] = calculate_rsi(hist['Close'])
            hist['Vol_Avg_20'] = hist['Volume'].rolling(window=20).mean()

            last = hist.iloc[-1]
            rsi = last.get('RSI_14')
            vol_avg = last.get('Vol_Avg_20')
            current_vol = last['Volume']

            if pd.isnull(rsi) or pd.isnull(vol_avg) or vol_avg == 0:
                continue

            vol_ratio = current_vol / vol_avg

            # 조건: RSI < 35 또는 거래량 2배 이상
            if rsi < 35 or vol_ratio > 2.0:
                # 점수 계산: RSI 낮을수록 + 거래량 높을수록 높은 점수
                score = 0
                if rsi < 30:
                    score += 3
                elif rsi < 35:
                    score += 2
                if vol_ratio > 3.0:
                    score += 3
                elif vol_ratio > 2.0:
                    score += 2
                elif vol_ratio > 1.5:
                    score += 1

                prev = hist.iloc[-2]
                change_pct = ((last['Close'] - prev['Close']) / prev['Close']) * 100

                name = TICKER_NAMES.get(ticker, ticker)
                gems.append({
                    "name": name,
                    "ticker": ticker,
                    "price": round(float(last['Close']), 0),
                    "change_pct": round(float(change_pct), 2),
                    "rsi_14": round(float(rsi), 2),
                    "vol_ratio": round(float(vol_ratio), 2),
                    "score": score,
                    "reason": _build_reason(rsi, vol_ratio),
                })
        except Exception as e:
            print(f"[SCAN] {ticker} skip: {e}", file=sys.stderr)
            continue

    gems.sort(key=lambda x: x['score'], reverse=True)
    return gems[:top_n]


def _build_reason(rsi, vol_ratio):
    reasons = []
    if rsi < 30:
        reasons.append(f"RSI {rsi:.1f} 과매도")
    elif rsi < 35:
        reasons.append(f"RSI {rsi:.1f} 과매도 근접")
    if vol_ratio > 2.0:
        reasons.append(f"거래량 {vol_ratio:.1f}배 급증")
    return " + ".join(reasons)
