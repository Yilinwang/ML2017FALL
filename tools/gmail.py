from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
import smtplib


def email_message(email_config, subject, email_data=None, content=''):
    '''
    if email_data is None, a pure text email is sent, otherwise an email is sent with attachments.

    args:
        email_config: dict with keys ('from', 'to', 'password', 'mail_server')
        subject: subject of the email
        email_data: attachments to send, list of dict with keys ('name', 'data')
        content: str, content of the email
    '''
    mail_server = login_mail_server(email_config)
    email_message = gen_message(email_config, subject, content)
    for data in email_data or []:
        attachment = gen_attachment(data)
        email_message.attach(attachment)
    mail_server.send_message(email_message)
    mail_server.close()


def gen_message(email_config, subject, content):
    message = MIMEMultipart()
    message['From'] = email_config['from']
    message['To'] = email_config['to']
    message['Date'] = formatdate(localtime=True)
    message['Subject'] = subject
    message.attach(MIMEText(content))
    return message


def gen_attachment(data):
    attachment = MIMEBase('application', "octet-stream")
    attachment.set_payload(data['data'].encode('utf8'))
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f"attachment; filename={data['name']}")
    return attachment


def login_mail_server(email_config):
    mail_server = smtplib.SMTP(email_config['mail_server'])
    mail_server.ehlo()
    mail_server.starttls()
    mail_server.login(email_config['from'], email_config['password'])
    return mail_server


def test():
    email_config = {
        'from': 'yentingg.lee@gmail.com',
        'to': 'ralph831005@gmail.com',
        'password': 'PASSWORD',
        'mail_server': 'smtp.gmail.com:587',
    }
    email_message(email_config, 'test', content='hello', email_data=[{'name': 'test.txt', 'data': 'hi'}])
    email_message(email_config, 'test2', content='hello world')


if __name__ == '__main__':
    test()
