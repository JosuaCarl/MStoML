# imports
import os
import mat73
import pandas as pd
import smtplib, ssl, rsa

def parse_folder(path):
    return os.listdir(path)


def mat_to_tsv(folder, file):
    """
    path: path to mat file
    saves mat files as tsv in the same folder
    """
    mat = mat73.loadmat(f"{folder}/{file}")
    for k, v in mat.items():
        if not os.path.isfile(f"{folder}/{k}.tsv"):
            df = pd.DataFrame(v)
            df.to_csv(f"{folder}/{k}.tsv", sep="\t", index=False)


def send_mail(subject, message):
    port = 587    # For SSL
    smtp_server = "smtpserv.uni-tuebingen.de"
    sender_email = "josua.carl@student.uni-tuebingen.de"  # Enter your address
    receiver_email = "josua.carl@student.uni-tuebingen.de"  # Enter receiver address
    with open(r"C:\Users\JosuaCarl\Desktop\pw.txt", "rb") as pw:
        with open(r"C:\Users\JosuaCarl\Desktop\privatekey.pem", "rb") as f:
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