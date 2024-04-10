from GPUtil import showUtilization, getAvailable
import psutil

def bits_to_bytes(bits, factor):
    """
    Coverts a number of bits to a number of bytes

    Args:
        bits: bits to be converted
        factor: / 10**factor (e.g. use 9 for GB)
    """
    return round((bits * 0.125) / 10**factor, 5)


def print_available_gpus():
    print(f"Available GPUs: {getAvailable(limit=4)}")


def print_utilization(gpu:bool=False):
    """
    Print the GPU, CPU and RAM utilization at the moment.
    """
    if gpu:
        showUtilization(all=True)
    print(f"CPU: {psutil.cpu_percent()}%")
    vm = psutil.virtual_memory()
    print(f"RAM: {bits_to_bytes(vm.used, 9)}GB / {bits_to_bytes(vm.total, 9)}GB ({vm.percent}%)")