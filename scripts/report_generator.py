"""HTML 리포트 생성."""

import json
import sys
import os
from datetime import datetime, timezone, timedelta

from analyzer import analyze_watchlist
from gem_scanner import scan_gems
from market_overview import get_market_indices, get_exchange_rates, get_vix, get_sector_performance
from extras import check_52week_alerts, get_weekly_performance, get_earnings_calendar
from sort_utils import sort_by_market_and_cap, fetch_market_caps
from ai_forecast import generate_ai_forecast_section


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
    "recovering": "회복 중",
    "normal": "정상",
    "uptrend": "상승",
    "downtrend": "하락",
    "sideways": "횡보",
    "golden_cross": "골든크로스",
    "dead_cross": "데드크로스",
    "undervalued": "저평가",
    "overvalued": "고평가",
    "none": "-",
}

REC_COLOR = {
    "strong_buy": "#059669",
    "buy": "#34D399",
    "hold": "#D4A853",
    "sell": "#F87171",
    "strong_sell": "#C0392B",
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
        return f'<span style="color:#DC2626">+{change_pct:.2f}%</span>'
    elif change_pct < 0:
        return f'<span style="color:#2563EB">{change_pct:.2f}%</span>'
    return f'<span style="color:#6B7280">{change_pct:.2f}%</span>'


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

    # 시총 기반 정렬 (KRX 먼저, 각 그룹 내 시총 내림차순)
    print("=== Sorting by Market Cap ===", file=sys.stderr)
    ticker_info_cache = fetch_market_caps(results)
    results = sort_by_market_and_cap(results, ticker_info_cache)

    krx_stocks = [r for r in results if r['market'] == 'KRX']
    us_stocks = [r for r in results if r['market'] == 'US']

    # 2.5. AI 예측 (Kronos)
    print("=== AI Forecast ===", file=sys.stderr)
    try:
        ai_forecast_html = generate_ai_forecast_section(
            results, ticker_info_cache, timeout_seconds=600
        )
    except Exception as e:
        print(f"AI 예측 섹션 생성 실패: {e}", file=sys.stderr)
        ai_forecast_html = ""

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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ margin: 0; padding: 0; background: #F1EDE8; color: #1A1A1A; font-family: 'Georgia', 'Times New Roman', serif; -webkit-font-smoothing: antialiased; }}
  .wrapper {{ max-width: 680px; margin: 0 auto; padding: 40px 20px; }}
  .header {{ text-align: center; padding: 48px 32px 40px; border-bottom: 1px solid #D6CFC7; }}
  .header-label {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #9B8E7E; margin-bottom: 16px; }}
  .header-title {{ font-size: 32px; font-weight: 400; color: #1A1A1A; letter-spacing: -0.5px; margin: 0; }}
  .header-date {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 12px; color: #9B8E7E; margin-top: 12px; letter-spacing: 0.5px; }}
  .section {{ padding: 32px 0; border-bottom: 1px solid #E8E2DB; }}
  .section:last-child {{ border-bottom: none; }}
  .section-title {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 11px; letter-spacing: 2.5px; text-transform: uppercase; color: #9B8E7E; margin: 0 0 24px 0; font-weight: 600; }}
  .section-note {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; color: #9B8E7E; font-size: 12px; margin: -16px 0 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0; }}
  th {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; background: #2C2620; color: #E8E2DB; padding: 10px 8px; text-align: left; font-size: 11px; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }}
  th:first-child {{ border-radius: 6px 0 0 0; }}
  th:last-child {{ border-radius: 0 6px 0 0; }}
  td {{ padding: 10px 8px; border-bottom: 1px solid #E8E2DB; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 13px; color: #3D3529; }}
  tr:last-child td {{ border-bottom: none; }}
  .idx-grid {{ width: 100%; }}
  .idx-grid td {{ background: #FDFBF8; border: 4px solid #F1EDE8; border-radius: 8px; padding: 16px 20px; text-align: center; width: 20%; }}
  .idx-label {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: #9B8E7E; }}
  .idx-value {{ font-size: 20px; font-weight: 700; color: #1A1A1A; margin: 6px 0 4px; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; }}
  .idx-change {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 12px; }}
  .rec {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; padding: 3px 10px; border-radius: 20px; color: white; font-weight: 600; font-size: 11px; letter-spacing: 0.3px; white-space: nowrap; }}
  .card {{ padding: 20px 24px; margin: 10px 0; border-radius: 10px; border: 1px solid #E8E2DB; }}
  .card-alert-high {{ background: linear-gradient(135deg, #FDF2F2 0%, #FDFBF8 100%); border-color: #F5D5D5; }}
  .card-alert-low {{ background: linear-gradient(135deg, #EEF2FF 0%, #FDFBF8 100%); border-color: #D5DCFA; }}
  .card-earnings {{ background: linear-gradient(135deg, #F5F0FF 0%, #FDFBF8 100%); border-color: #DDD5F5; }}
  .card-gem {{ background: linear-gradient(135deg, #FDF8EE 0%, #FDFBF8 100%); border-color: #EDE3CC; }}
  .card-title {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-weight: 600; font-size: 14px; color: #1A1A1A; }}
  .card-sub {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 12px; color: #9B8E7E; margin-top: 4px; }}
  .card-pip {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }}
  .card-reason {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; color: #B8860B; font-size: 12px; margin-top: 6px; font-weight: 500; }}
  .vix-badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 10px; font-weight: 600; color: white; letter-spacing: 0.5px; vertical-align: middle; }}
  .sector-bar {{ height: 4px; border-radius: 2px; margin-top: 6px; }}
  .ai-table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; }}
  .ai-table th {{ background: #2C2620; color: #E8E2DB; padding: 9px 8px; text-align: left; font-size: 11px; font-weight: 500; letter-spacing: 0.5px; }}
  .ai-table td {{ padding: 9px 8px; border-bottom: 1px solid #E8E2DB; color: #3D3529; font-size: 13px; }}
  .ai-name {{ font-weight: 600; }}
  .ai-ticker {{ color: #9B8E7E; font-size: 10px; }}
  .ai-skip {{ color: #9B8E7E; font-size: 12px; }}
  .ai-dir {{ text-align: center; }}
  .ai-row-alt {{ background: #FDFBF8; }}
  .ai-hot {{ display: inline-block; background: #D4A853; color: #FDFBF8; font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 3px; margin-left: 5px; vertical-align: middle; letter-spacing: 0.3px; }}
  .ai-wrap {{ border: 1px solid #E8E2DB; border-radius: 10px; padding: 20px 24px; margin: 0; }}
  .ai-sub {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: #9B8E7E; margin: 0 0 10px 0; font-weight: 600; }}
  .footer {{ padding: 40px 0 20px; text-align: center; }}
  .footer-line {{ width: 40px; height: 1px; background: #D6CFC7; margin: 0 auto 20px; }}
  .footer p {{ font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; color: #B8AFA3; font-size: 11px; line-height: 1.8; margin: 0; }}
</style>
</head>
<body>
<div class="wrapper">
<div class="header">
  <div class="header-label">Daily Briefing</div>
  <h1 class="header-title">Stock Report</h1>
  <div class="header-date">{now} KST</div>
</div>
"""

    # === 시장 요약 ===
    html += '<div class="section">'
    html += '<div class="section-title">Market Overview</div>'

    # 지수 그리드 (테이블 기반 — 이메일 호환)
    html += '<table class="idx-grid" cellspacing="0" cellpadding="0"><tr>'
    for idx in indices:
        html += f"""<td>
  <div class="idx-label">{idx['name']}</div>
  <div class="idx-value">{idx['value']:,.2f}</div>
  <div class="idx-change">{format_change(idx['change_pct'])}</div>
</td>"""
    html += '</tr></table>'

    # 환율 + VIX
    fx_count = len(fx_rates) + (1 if vix else 0)
    html += '<table class="idx-grid" cellspacing="0" cellpadding="0" style="margin-top:4px"><tr>'

    for r in fx_rates:
        html += f"""<td>
  <div class="idx-label">{r['name']}</div>
  <div class="idx-value">₩{r['value']:,.2f}</div>
  <div class="idx-change">{format_change(r['change_pct'])}</div>
</td>"""

    if vix:
        vix_color = "#C0392B" if vix['value'] >= 25 else "#D4A853" if vix['value'] >= 18 else "#059669"
        html += f"""<td>
  <div class="idx-label">VIX</div>
  <div class="idx-value">{vix['value']:.2f}</div>
  <div class="idx-change"><span class="vix-badge" style="background:{vix_color}">{vix['level']}</span></div>
</td>"""

    html += '</tr></table>'
    html += '</div>'

    # === 섹터 흐름 ===
    if sectors:
        html += '<div class="section">'
        html += '<div class="section-title">Sector Performance</div>'
        html += '<table><tr><th>섹터</th><th>등락률</th><th style="width:40%"></th></tr>'
        for s in sectors:
            bar_color = "#059669" if s['change_pct'] >= 0 else "#DC2626"
            bar_width = min(abs(s['change_pct']) * 20, 100)
            html += f"""<tr>
  <td style="font-weight:600">{s['name']}</td>
  <td>{format_change(s['change_pct'])}</td>
  <td><div class="sector-bar" style="background:{bar_color};width:{bar_width}%"></div></td>
</tr>"""
        html += '</table>'
        html += '</div>'

    # === 52주 신고가/신저가 알림 ===
    if alerts_52w:
        html += '<div class="section">'
        html += '<div class="section-title">52-Week Alerts</div>'
        html += '<p class="section-note">관심종목 중 52주 최고/최저 가격 근접 종목</p>'
        for a in alerts_52w:
            css_class = "card-alert-high" if a['is_high'] else "card-alert-low"
            pip_color = "#C0392B" if a['is_high'] else "#2563EB"
            html += f"""<div class="card {css_class}">
  <div class="card-title"><span class="card-pip" style="background:{pip_color}"></span>{a['name']} <span style="color:#9B8E7E;font-weight:400">({a['ticker']})</span> — {format_price(a['price'], a['currency'])}</div>
  <div class="card-sub">{a['alert_type']} &middot; 52주 고가: {format_price(a['high_52w'], a['currency'])} / 저가: {format_price(a['low_52w'], a['currency'])}</div>
</div>"""
        html += '</div>'

    # === 실적 발표 캘린더 ===
    if earnings:
        html += '<div class="section">'
        html += '<div class="section-title">Earnings Calendar</div>'
        for e in earnings:
            html += f"""<div class="card card-earnings">
  <div class="card-title"><span class="card-pip" style="background:#7C3AED"></span>{e['name']} <span style="color:#9B8E7E;font-weight:400">({e['ticker']})</span></div>
  <div class="card-sub">{e['date']}</div>
</div>"""
        html += '</div>'

    # === 주간 성과 (월요일만) ===
    if weekly:
        html += '<div class="section">'
        html += '<div class="section-title">Weekly Performance</div>'
        html += '<table><tr><th>종목</th><th>주간 등락률</th></tr>'
        for w in weekly:
            html += f'<tr><td style="font-weight:600">{w["name"]}</td><td>{format_change(w["week_change_pct"])}</td></tr>'
        html += '</table>'
        html += '</div>'

    # === 관심종목 분석 ===
    if krx_stocks:
        html += '<div class="section">'
        html += _build_stock_table("Korean Equities", "KRX", krx_stocks)
        html += '</div>'

    if us_stocks:
        html += '<div class="section">'
        html += _build_stock_table("US Equities", "US", us_stocks)
        html += '</div>'

    # === AI 예측 (Kronos) ===
    if ai_forecast_html:
        html += ai_forecast_html

    # === 숨겨진 종목 추천 ===
    html += '<div class="section">'
    html += '<div class="section-title">Hidden Gems</div>'
    html += '<p class="section-note">KOSPI/KOSDAQ 중 거래량 급증 또는 RSI 과매도 종목</p>'

    if gems:
        for g in gems:
            color = "#059669" if g['change_pct'] >= 0 else "#2563EB"
            html += f"""<div class="card card-gem">
  <div class="card-title"><span class="card-pip" style="background:#D4A853"></span>{g['name']} <span style="color:#9B8E7E;font-weight:400">({g['ticker']})</span>
    <span style="margin-left:8px;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif">₩{g['price']:,.0f}</span>
    <span style="color:{color};margin-left:4px;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif">{g['change_pct']:+.2f}%</span>
  </div>
  <div class="card-reason">{g['reason']}</div>
</div>"""
    else:
        html += '<p style="color:#9B8E7E;font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:13px">오늘은 조건에 부합하는 종목이 없습니다.</p>'

    html += '</div>'

    html += """
<div class="footer">
  <div class="footer-line"></div>
  <p>이 리포트는 과거 데이터 기반 기술적 분석이며,<br>투자 판단의 최종 책임은 본인에게 있습니다.</p>
  <p style="margin-top:8px">Yahoo Finance &middot; daily-stock-report</p>
  <p style="margin-top:12px"><a href="https://github.com/jisub-kim" style="color:#B8AFA3;text-decoration:none;border-bottom:1px solid #D6CFC7">jisub-kim</a></p>
</div>
</div>
</body>
</html>"""

    return html


def _chart_url(ticker, market):
    if market == 'KRX':
        code = ticker.split('.')[0]
        return f"https://finance.naver.com/item/fchart.naver?code={code}"
    return f"https://www.tradingview.com/chart/?symbol={ticker}"


def _build_stock_table(title, market_label, stocks):
    html = f'<div class="section-title">{title}</div>'
    html += """<table>
<tr>
  <th>종목</th><th>현재가</th><th>등락률</th>
  <th>RSI</th><th>MACD</th><th>추세</th><th>단기</th>
  <th>PER</th><th>PBR</th><th>배당률</th><th>크로스</th><th>장기</th>
</tr>"""

    for i, s in enumerate(stocks):
        sig = s['signals']
        rec = sig['recommendation']
        rec_kr = SIGNAL_KR.get(rec, rec)
        rec_color = REC_COLOR.get(rec, "#9B8E7E")
        trend_kr = SIGNAL_KR.get(sig['trend'], sig['trend'])
        macd_kr = SIGNAL_KR.get(sig['macd_signal'], sig['macd_signal'])

        lt = s.get('long_term', {})
        lt_rec = lt.get('recommendation', 'hold')
        lt_rec_kr = SIGNAL_KR.get(lt_rec, lt_rec)
        lt_rec_color = REC_COLOR.get(lt_rec, "#9B8E7E")
        cross_kr = SIGNAL_KR.get(lt.get('cross', 'none'), '-')

        per_str = f"{s['per']:.1f}" if s.get('per') else '-'
        pbr_str = f"{s['pbr']:.2f}" if s.get('pbr') else '-'
        div_str = f"{s['dividend_yield']:.1f}%" if s.get('dividend_yield') else '-'

        rsi_color = ""
        if s['rsi_14'] and s['rsi_14'] > 70:
            rsi_color = 'color:#DC2626'
        elif s['rsi_14'] and s['rsi_14'] < 30:
            rsi_color = 'color:#2563EB'

        row_bg = 'background:#FDFBF8;' if i % 2 == 0 else ''

        html += f"""<tr style="{row_bg}">
  <td><a href="{_chart_url(s['ticker'], s['market'])}" style="color:#1A1A1A;text-decoration:none;font-weight:600">{s['name']}</a><br><span style="color:#9B8E7E;font-size:10px;letter-spacing:0.3px">{s['ticker']}</span></td>
  <td style="font-weight:600">{format_price(s['price'], s['currency'])}</td>
  <td>{format_change(s['change_pct'])}</td>
  <td style="{rsi_color};font-weight:500">{s['rsi_14'] or '-'}</td>
  <td>{macd_kr}</td>
  <td>{trend_kr}</td>
  <td><span class="rec" style="background:{rec_color}">{rec_kr}</span></td>
  <td style="color:#6B7280">{per_str}</td>
  <td style="color:#6B7280">{pbr_str}</td>
  <td style="color:#6B7280">{div_str}</td>
  <td>{cross_kr}</td>
  <td><span class="rec" style="background:{lt_rec_color}">{lt_rec_kr}</span></td>
</tr>"""

    html += "</table>"
    return html


if __name__ == "__main__":
    watchlist_path = sys.argv[1] if len(sys.argv) > 1 else "watchlist.json"
    html = generate_html(watchlist_path)
    print(html)
