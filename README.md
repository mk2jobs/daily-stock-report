# Daily Stock Report

매일 아침 KST 08:00에 관심종목 기술적 분석 + 숨겨진 종목 추천 리포트를 이메일로 발송합니다.

## 기능

- **관심종목 분석**: RSI, MACD, 볼린저 밴드, 이동평균선 기반 매매 신호
- **숨겨진 종목 추천**: KOSPI/KOSDAQ 중 거래량 급증 + RSI 과매도 종목 Top 10
- **HTML 리포트**: 깔끔한 테이블 형식으로 Gmail 발송
- **멀티마켓**: 한국(KRX), 미국(US) 주식 동시 지원

## 종목 관리

`watchlist.json`을 편집하면 됩니다:

```json
{
  "stocks": [
    {"name": "삼성전자", "ticker": "005930.KS", "market": "KRX"},
    {"name": "애플", "ticker": "AAPL", "market": "US"}
  ]
}
```

### 티커 형식

| 시장 | 형식 | 예시 |
|---|---|---|
| KOSPI | 종목코드.KS | 005930.KS |
| KOSDAQ | 종목코드.KQ | 035900.KQ |
| US | 심볼 | AAPL, MSFT |

## GitHub Secrets 설정

| Secret | 값 |
|---|---|
| `GMAIL_USER` | 발신 Gmail 주소 |
| `GMAIL_APP_PASSWORD` | Gmail 앱 비밀번호 (16자리) |
| `REPORT_EMAIL` | 수신 이메일 주소 |

## 수동 실행

Actions 탭 → Daily Stock Report → Run workflow

## 로컬 테스트

```bash
pip install -r requirements.txt
cd scripts
python report_generator.py ../watchlist.json > report.html
open report.html
```
