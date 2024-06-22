import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# https://myaccount.google.com/apppasswords
# Navigate to App Password Generator, designate an app name such as "security project," and obtain a 16-digit password.
# Copy this password and paste it into the designated password field as instructed.

password = ""
from_email = ""  # must match the email used to generate the password
to_email = ""  # receiver email

server = smtplib.SMTP("smtp.gmail.com: 587")
server.starttls()
server.login(from_email, password)


def send_email(to_email, from_email, object_detected=1):
    """Sends an email notification indicating the number of objects detected; defaults to 1 object."""
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"
    # Add in the message body
    message_body = f"ALERT - {object_detected} objects has been detected!!"

    message.attach(MIMEText(message_body, "plain"))
    server.sendmail(from_email, to_email, message.as_string())
