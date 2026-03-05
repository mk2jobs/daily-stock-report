"""HTML 리포트 생성."""

import json
import sys
import os
from datetime import datetime, timezone, timedelta

from analyzer import analyze_watchlist
from gem_scanner import scan_gems
from market_overview import get_market_indices, get_exchange_rates, get_vix, get_sector_performance
from extras import check_52week_alerts, get_weekly_performance, get_earnings_calendar


SIGNAL_KR = {
    "strong_buy": "강력 매수",
    "buy": "매수",
    "hold": "보유",
    "sell": "매도",
    "strong_sell": "강력 매도",
    "bullish": "강세",
    "bearish": "약세",
    "neutral": "중립",
    "overbought": "과매수",
    "oversold": "과매도",
    "normal": "정상",
    "uptrend": "상승",
    "downtrend": "하락",
    "sideways": "횡보",
}

REC_COLOR = {
    "strong_buy": "#00C853",
    "buy": "#4CAF50",
    "hold": "#FF9800",
    "sell": "#F44336",
    "strong_sell": "#B71C1C",
}


def format_price(price, currency):
    if currency == "KRW":
        return f"₩{price:,.0f}"
    elif currency == "HKD":
        return f"HK${price:,.2f}"
    elif currency == "CNY":
        return f"¥{price:,.2f}"
    return f"${price:,.2f}"


def format_change(change_pct):
    if change_pct > 0:
        return f'<span style="color:#F44336">+{change_pct:.2f}%</span>'
    elif change_pct < 0:
        return f'<span style="color:#2196F3">{change_pct:.2f}%</span>'
    return f'{change_pct:.2f}%'


def generate_html(watchlist_path):
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst).strftime("%Y-%m-%d %H:%M")

    # 1. 시장 요약
    print("=== Market Overview ===", file=sys.stderr)
    indices = get_market_indices()
    fx_rates = get_exchange_rates()
    vix = get_vix()
    sectors = get_sector_performance()

    # 2. 관심종목 분석
    print("=== Watchlist Analysis ===", file=sys.stderr)
    results = analyze_watchlist(watchlist_path)

    krx_stocks = [r for r in results if r['market'] == 'KRX']
    us_stocks = [r for r in results if r['market'] == 'US']

    # 3. 추가 분석
    print("=== Extras ===", file=sys.stderr)
    alerts_52w = check_52week_alerts(results)
    weekly = get_weekly_performance(results)
    earnings = get_earnings_calendar(results)

    # 4. 숨겨진 종목 스캔
    print("=== Gem Scanner ===", file=sys.stderr)
    gems = scan_gems(top_n=10)

    # HTML 생성
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; color: #333; }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  h3 {{ color: #333; margin-top: 20px; font-size: 15px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #16213e; color: white; padding: 12px 10px; text-align: left; font-size: 13px; }}
  td {{ padding: 10px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tr:hover {{ background: #f8f9fa; }}
  .rec {{ padding: 3px 8px; border-radius: 4px; color: white; font-weight: bold; font-size: 12px; }}
  .gem-card {{ background: white; padding: 12px 16px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #FF9800; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .gem-name {{ font-weight: bold; font-size: 15px; }}
  .gem-reason {{ color: #e65100; font-size: 13px; margin-top: 4px; }}
  .alert-card {{ background: white; padding: 12px 16px; margin: 8px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .alert-high {{ border-left: 4px solid #F44336; }}
  .alert-low {{ border-left: 4px solid #2196F3; }}
  .overview-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }}
  .overview-card {{ background: white; padding: 14px 18px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 140px; }}
  .overview-label {{ font-size: 12px; color: #999; }}
  .overview-value {{ font-size: 18px; font-weight: bold; margin-top: 4px; }}
  .overview-change {{ font-size: 13px; margin-top: 2px; }}
  .vix-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white; }}
  .sector-bar {{ height: 8px; border-radius: 4px; margin-top: 4px; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
  .section-note {{ color: #666; font-size: 13px; margin-bottom: 10px; }}
  .earnings-item {{ background: white; padding: 10px 16px; margin: 6px 0; border-radius: 8px; border-left: 4px solid #9C27B0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<div class="container">
<h1>Daily Stock Report</h1>
<p style="color:#666">{now} KST</p>
"""

    # === 시장 요약 ===
    html += '<h2>시장 요약</h2>'
    html += '<div class="overview-grid">'

    for idx in indices:
        html += f"""<div class="overview-card">
  <div class="overview-label">{idx['name']}</div>
  <div class="overview-value">{idx['value']:,.2f}</div>
  <div class="overview-change">{format_change(idx['change_pct'])}</div>
</div>"""

    html += '</div>'

    # 환율 + VIX
    html += '<div class="overview-grid">'

    for r in fx_rates:
        html += f"""<div class="overview-card">
  <div class="overview-label">{r['name']}</div>
  <div class="overview-value">₩{r['value']:,.2f}</div>
  <div class="overview-change">{format_change(r['change_pct'])}</div>
</div>"""

    if vix:
        vix_color = "#F44336" if vix['value'] >= 25 else "#FF9800" if vix['value'] >= 18 else "#4CAF50"
        html += f"""<div class="overview-card">
  <div class="overview-label">VIX (공포지수)</div>
  <div class="overview-value">{vix['value']:.2f} <span class="vix-badge" style="background:{vix_color}">{vix['level']}</span></div>
  <div class="overview-change">{format_change(vix['change_pct'])}</div>
</div>"""

    html += '</div>'

    # === 섹터 흐름 ===
    if sectors:
        html += '<h3>섹터 흐름 (미국 ETF 기준)</h3>'
        html += '<table><tr><th>섹터</th><th>등락률</th><th></th></tr>'
        for s in sectors:
            bar_color = "#4CAF50" if s['change_pct'] >= 0 else "#F44336"
            bar_width = min(abs(s['change_pct']) * 20, 100)
            html += f"""<tr>
  <td><strong>{s['name']}</strong></td>
  <td>{format_change(s['change_pct'])}</td>
  <td><div class="sector-bar" style="background:{bar_color};width:{bar_width}%"></div></td>
</tr>"""
        html += '</table>'

    # === 52주 신고가/신저가 알림 ===
    if alerts_52w:
        html += '<h2>52주 신고가/신저가 알림</h2>'
        html += '<p class="section-note">관심종목 중 52주 최고/최저 가격 근접 종목</p>'
        for a in alerts_52w:
            css_class = "alert-high" if a['is_high'] else "alert-low"
            icon = "📈" if a['is_high'] else "📉"
            html += f"""<div class="alert-card {css_class}">
  <strong>{icon} {a['name']}</strong> <span style="color:#999">({a['ticker']})</span>
  — {format_price(a['price'], a['currency'])}
  <br><span style="font-size:12px;color:#666">{a['alert_type']} | 52주 고가: {format_price(a['high_52w'], a['currency'])} / 저가: {format_price(a['low_52w'], a['currency'])}</span>
</div>"""

    # === 실적 발표 캘린더 ===
    if earnings:
        html += '<h2>이번 주 실적 발표</h2>'
        for e in earnings:
            html += f"""<div class="earnings-item">
  <strong>{e['name']}</strong> <span style="color:#999">({e['ticker']})</span>
  — {e['date']}
</div>"""

    # === 주간 성과 (월요일만) ===
    if weekly:
        html += '<h2>지난주 성과 요약</h2>'
        html += '<table><tr><th>종목</th><th>주간 등락률</th></tr>'
        for w in weekly:
            html += f'<tr><td><strong>{w["name"]}</strong></td><td>{format_change(w["week_change_pct"])}</td></tr>'
        html += '</table>'

    # === 관심종목 분석 ===
    if krx_stocks:
        html += _build_stock_table("한국 주식 (KRX)", krx_stocks)

    if us_stocks:
        html += _build_stock_table("미국 주식 (US)", us_stocks)

    # === 숨겨진 종목 추천 ===
    html += '<h2>숨겨진 종목 추천</h2>'
    html += '<p class="section-note">KOSPI/KOSDAQ 중 거래량 급증 또는 RSI 과매도 종목</p>'

    if gems:
        for g in gems:
            color = "#4CAF50" if g['change_pct'] >= 0 else "#2196F3"
            html += f"""<div class="gem-card">
  <span class="gem-name">{g['name']}</span> <span style="color:#999">({g['ticker']})</span>
  <span style="margin-left:10px">₩{g['price']:,.0f}</span>
  <span style="color:{color}; margin-left:5px">{g['change_pct']:+.2f}%</span>
  <div class="gem-reason">{g['reason']}</div>
</div>"""
    else:
        html += '<p style="color:#999">오늘은 조건에 부합하는 종목이 없습니다.</p>'

    html += """
<div class="footer">
  <p>이 리포트는 과거 데이터 기반 기술적 분석이며, 투자 판단의 최종 책임은 본인에게 있습니다.</p>
  <p>Data source: Yahoo Finance (yfinance) | Generated by daily-stock-report</p>
</div>
</div>
</body>
</html>"""

    return html


def _build_stock_table(title, stocks):
    html = f'<h2>{title}</h2>'
    html += """<table>
<tr>
  <th>종목</th><th>현재가</th><th>등락률</th>
  <th>RSI</th><th>MACD</th><th>추세</th><th>추천</th>
</tr>"""

    for s in stocks:
        sig = s['signals']
        rec = sig['recommendation']
        rec_kr = SIGNAL_KR.get(rec, rec)
        rec_color = REC_COLOR.get(rec, "#999")
        trend_kr = SIGNAL_KR.get(sig['trend'], sig['trend'])
        macd_kr = SIGNAL_KR.get(sig['macd_signal'], sig['macd_signal'])

        rsi_color = ""
        if s['rsi_14'] and s['rsi_14'] > 70:
            rsi_color = 'color:#F44336'
        elif s['rsi_14'] and s['rsi_14'] < 30:
            rsi_color = 'color:#2196F3'

        html += f"""<tr>
  <td><strong>{s['name']}</strong><br><span style="color:#999;font-size:11px">{s['ticker']}</span></td>
  <td>{format_price(s['price'], s['currency'])}</td>
  <td>{format_change(s['change_pct'])}</td>
  <td style="{rsi_color}">{s['rsi_14'] or '-'}</td>
  <td>{macd_kr}</td>
  <td>{trend_kr}</td>
  <td><span class="rec" style="background:{rec_color}">{rec_kr}</span></td>
</tr>"""

    html += "</table>"
    return html


if __name__ == "__main__":
    watchlist_path = sys.argv[1] if len(sys.argv) > 1 else "watchlist.json"
    html = generate_html(watchlist_path)
    print(html)
