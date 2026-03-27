"""Gmail SMTP로 리포트 발송."""

import smtplib
import sys
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_report(html_content, to_emails, gmail_user, gmail_app_password):
    recipients = [e.strip() for e in to_emails.split(',')]

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"[Daily Stock Report] {datetime.now().strftime('%Y-%m-%d')} 일일 리포트"
    msg['From'] = gmail_user
    msg['To'] = ', '.join(recipients)

    html_part = MIMEText(html_content, 'html', 'utf-8')
    msg.attach(html_part)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(gmail_user, gmail_app_password)
        server.sendmail(gmail_user, recipients, msg.as_string())

    print(f"Report sent to {', '.join(recipients)}", file=sys.stderr)


if __name__ == "__main__":
    html = sys.stdin.read()
    to_emails = os.environ['REPORT_EMAIL']
    gmail_user = os.environ['GMAIL_USER']
    gmail_app_password = os.environ['GMAIL_APP_PASSWORD']

    send_report(html, to_emails, gmail_user, gmail_app_password)
