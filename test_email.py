import smtplib

EMAIL    = "lenkagudu4@gmail.com"
PASSWORD = "murualognssxeusd"        # paste your app password here
TO       = "subhamlenka021@gmail.com"

try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(EMAIL, PASSWORD)
    server.sendmail(EMAIL, TO, "Subject: NALCO Test\n\nThis is a test email from NALCO HEMM!")
    server.quit()
    print("SUCCESS — Email sent!")
except Exception as e:
    print("FAILED:", e)