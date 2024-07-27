from .notification_sender import NotificationSender
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# https://myaccount.google.com/apppasswords
# Navigate to App Password Generator, designate an app name such as "security project," and obtain a 16-digit password.
# Copy this password and paste it into the designated password field as instructed.


class MailNotificationSender(NotificationSender):

    def __init__(self, from_email: str, to_email: str, password: str):
        self.server = smtplib.SMTP("smtp.gmail.com: 587")
        self.server.starttls()
        self.server.login(from_email, password)
        # must match the email used to generate the password
        self.from_email = from_email
        # receiver email
        self.to_email = to_email

    def send(self, payload: any) -> bool:
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"
        # TODO: ensure payload is a string
        # message_body = f"ALERT - {object_detected} objects has been detected!!"
        message.attach(MIMEText(payload, "plain"))
        try:
            self.server.sendmail(self.from_email, self.to_email, message.as_string())
        except Exception as e:
            print(f"[DEBUG] Failed to send mail notification: {e}")
            return False
        return True
