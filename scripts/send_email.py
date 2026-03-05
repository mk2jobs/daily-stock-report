"""Gmail SMTP로 리포트 발송."""

import smtplib
import sys
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_report(html_content, to_email, gmail_user, gmail_app_password):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"[Daily Stock Report] {datetime.now().strftime('%Y-%m-%d')} 일일 리포트"
    msg['From'] = gmail_user
    msg['To'] = to_email

    html_part = MIMEText(html_content, 'html', 'utf-8')
    msg.attach(html_part)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(gmail_user, gmail_app_password)
        server.sendmail(gmail_user, to_email, msg.as_string())

    print(f"Report sent to {to_email}", file=sys.stderr)


if __name__ == "__main__":
    html = sys.stdin.read()
    to_email = os.environ.get('REPORT_EMAIL', 'jisobkim@gmail.com')
    gmail_user = os.environ.get('GMAIL_USER', 'jisobkim@gmail.com')
    gmail_app_password = os.environ['GMAIL_APP_PASSWORD']

    send_report(html, to_email, gmail_user, gmail_app_password)
