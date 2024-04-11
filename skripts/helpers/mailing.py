import smtplib, ssl, rsa

def send_mail(subject, message, pw_file, private_key):
    port = 587    # For SSL
    smtp_server = "smtpserv.uni-tuebingen.de"
    sender_email = "josua.carl@student.uni-tuebingen.de"  # Enter your address
    receiver_email = "josua.carl@student.uni-tuebingen.de"  # Enter receiver address
    with open(pw_file, "rb") as pw:
        with open(private_key, "rb") as f:
            crypt = pw.read()
            key = rsa.PrivateKey.load_pkcs1(f.read())
            password = rsa.decrypt(crypt, key).decode()
    message = f"""\
    Subject:{subject}
    {message}"""

    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login("zxoeu03", password)
        server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()