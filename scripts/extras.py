"""추가 분석: 52주 신고가/신저가, 주간 성과, 실적 캘린더."""

import sys
from datetime import datetime, timezone, timedelta
import pandas as pd


def check_52week_alerts(watchlist_results):
    """관심종목 중 52주 최고/최저 근접 종목 탐지."""
    import yfinance as yf

    alerts = []
    for stock in watchlist_results:
        try:
            t = yf.Ticker(stock['ticker'])
            hist = t.history(period="1y", interval="1d")
            if hist.empty or len(hist) < 50:
                continue

            high_52w = float(hist['High'].max())
            low_52w = float(hist['Low'].min())
            current = stock['price']

            near_high = current >= high_52w * 0.95
            near_low = current <= low_52w * 1.05

            if near_high or near_low:
                alert_type = ""
                if current >= high_52w * 0.99:
                    alert_type = "52주 신고가"
                elif near_high:
                    alert_type = "52주 고가 근접 (5% 이내)"
                elif current <= low_52w * 1.01:
                    alert_type = "52주 신저가"
                elif near_low:
                    alert_type = "52주 저가 근접 (5% 이내)"

                alerts.append({
                    "name": stock['name'],
                    "ticker": stock['ticker'],
                    "price": current,
                    "currency": stock['currency'],
                    "high_52w": round(high_52w, 2),
                    "low_52w": round(low_52w, 2),
                    "alert_type": alert_type,
                    "is_high": near_high,
                })
        except Exception as e:
            print(f"[52W] {stock['ticker']} skip: {e}", file=sys.stderr)

    return alerts


def get_weekly_performance(watchlist_results):
    """관심종목 주간 등락률 (월요일 리포트용)."""
    import yfinance as yf

    kst = timezone(timedelta(hours=9))
    today = datetime.now(kst)

    # 월요일(0)에만 실행
    if today.weekday() != 0:
        return None

    results = []
    for stock in watchlist_results:
        try:
            t = yf.Ticker(stock['ticker'])
            hist = t.history(period="2wk", interval="1d")
            if hist.empty or len(hist) < 5:
                continue

            # 지난주 월요일 시가 vs 금요일 종가
            last_5 = hist.tail(6)  # 최근 6일 (이번주 월 + 지난주 5일)
            if len(last_5) >= 2:
                week_start = float(last_5.iloc[0]['Open'])
                week_end = float(last_5.iloc[-2]['Close'])  # 지난 금요일
                week_change = ((week_end - week_start) / week_start) * 100

                results.append({
                    "name": stock['name'],
                    "ticker": stock['ticker'],
                    "week_change_pct": round(float(week_change), 2),
                })
        except Exception as e:
            print(f"[WEEKLY] {stock['ticker']} skip: {e}", file=sys.stderr)

    results.sort(key=lambda x: x['week_change_pct'], reverse=True)
    return results


def get_earnings_calendar(watchlist_results):
    """관심종목 중 이번 주 실적 발표 예정 종목."""
    import yfinance as yf

    kst = timezone(timedelta(hours=9))
    today = datetime.now(kst).date()
    week_end = today + timedelta(days=7)

    earnings = []
    for stock in watchlist_results:
        try:
            t = yf.Ticker(stock['ticker'])
            cal = t.calendar
            if cal is None or cal.empty if hasattr(cal, 'empty') else not cal:
                continue

            # yfinance calendar는 dict 또는 DataFrame
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get('Earnings Date')
                if ed:
                    earnings_date = ed[0] if isinstance(ed, list) else ed
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.columns:
                    earnings_date = cal['Earnings Date'].iloc[0]

            if earnings_date is None:
                continue

            if hasattr(earnings_date, 'date'):
                ed = earnings_date.date()
            elif hasattr(earnings_date, 'to_pydatetime'):
                ed = earnings_date.to_pydatetime().date()
            else:
                continue

            if today <= ed <= week_end:
                earnings.append({
                    "name": stock['name'],
                    "ticker": stock['ticker'],
                    "date": ed.strftime("%m/%d (%a)"),
                })
        except Exception as e:
            print(f"[EARNINGS] {stock['ticker']} skip: {e}", file=sys.stderr)

    return earnings
