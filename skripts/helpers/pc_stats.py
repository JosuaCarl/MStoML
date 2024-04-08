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