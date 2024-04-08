# imports
import os
import mat73
import pandas as pd
import smtplib, ssl, rsa

from GPUtil import showUtilization
import psutil

def bits_to_bytes(bits, factor):
    """
    Coverts a number of bits to a number of bytes

    Args:
        bits: bits to be converted
        factor: / 10**factor (e.g. use 9 for GB)
    """
    return round((bits * 0.125) / 10**factor, 5)

def print_utilization():
    """
    Print the GPU, CPU and RAM utilization at the moment.
    """
    showUtilization(all=True)
    print(f"CPU: {psutil.cpu_percent()}%")
    vm = psutil.virtual_memory()
    print(f"RAM: {bits_to_bytes(vm.used, 9)}GB / {bits_to_bytes(vm.total, 9)}GB ({vm.percent}%)")

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

def mat_to_tsv_batch(folder:str):
    for file in parse_folder(folder):
        if file.endswith(".mat"):
            mat_to_tsv(folder, file)


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



# Pytorch size estimation
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class SizeEstimator(object):

    def __init__(self, model, input_size=(1,1,32,32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total