import numpy as np
import socket
import subprocess
from scipy.optimize import minimize
import time
import threading
import queue
import sys

# === CONFIG ===
SAMPLE_RATE = 250000
CHUNK_SIZE = 4096 #2048
LAMBDA_L1 = 0.141592653589793238462643383279502884197169399375105820974944592307816406286
EPSILON = 1e-3
DEFAULT_FREQ = 102000000
PORT = 9000

# Shared state
current_freq = DEFAULT_FREQ
proc = None
conn = None
freq_queue = queue.Queue()
terminate_flag = threading.Event()

def smooth_l1(x):
    return np.sqrt(x**2 + EPSILON**2)

def cost_function_factory(target):
    def cost(x):
        return 0.5 * np.sum((x - target)**2) + LAMBDA_L1 * np.sum(smooth_l1(x))
    return cost

def gradient_factory(target):
    def grad(x):
        return (x - target) + LAMBDA_L1 * x / np.sqrt(x**2 + EPSILON**2)
    return grad

def denoise_vector(data):
    x0 = np.zeros_like(data)
    result = minimize(
        cost_function_factory(data),
        x0,
        jac=gradient_factory(data),
        method="L-BFGS-B",
        options={"maxiter": 30}
    )
    return result.x

def start_rtl_sdr(freq_hz):
    print(f"Starting rtl_sdr at {freq_hz / 1e6:.3f} MHz")
    return subprocess.Popen(
        ['rtl_sdr', '-f', str(freq_hz), '-s', str(SAMPLE_RATE), '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7
    )

def start_tcp_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', port))
    s.listen(1)
    print(f"Waiting for SDR++ to connect on port {port}...")
    conn, addr = s.accept()
    print(f"SDR++ connected from {addr}")
    return conn

def frequency_input_loop():
    global current_freq
    while not terminate_flag.is_set():
        try:
            user_input = input("Enter new frequency in MHz (e.g. 145.800): ").strip()
            if not user_input:
                continue
            new_freq = int(float(user_input) * 1e6)
            freq_queue.put(new_freq)
        except Exception as e:
            print(f"Invalid input: {e}")

def iq_processing_loop():
    global proc, conn, current_freq

    proc = start_rtl_sdr(current_freq)
    conn = start_tcp_server(PORT)

    while not terminate_flag.is_set():
        # Check for frequency change
        try:
            while not freq_queue.empty():
                new_freq = freq_queue.get_nowait()
                if new_freq != current_freq:
                    print(f"Changing frequency to {new_freq / 1e6:.3f} MHz")
                    current_freq = new_freq
                    proc.kill()
                    proc.wait()
                    proc = start_rtl_sdr(current_freq)
        except queue.Empty:
            pass

        # Read and denoise
        raw = proc.stdout.read(CHUNK_SIZE * 2)
        if len(raw) < CHUNK_SIZE * 2:
            continue

        iq = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        iq = (iq - 127.5) / 127.5
        i = iq[::2]
        q = iq[1::2]

        i_d = denoise_vector(i)
        q_d = denoise_vector(q)
        iq_denoised = i_d + 1j * q_d

        try:
            conn.sendall(iq_denoised.astype(np.complex64).tobytes())
        except (BrokenPipeError, ConnectionResetError):
            print("SDR++ disconnected. Waiting for reconnect...")
            conn.close()
            conn = start_tcp_server(PORT)

def main():
    try:
        threading.Thread(target=frequency_input_loop, daemon=True).start()
        iq_processing_loop()
    except KeyboardInterrupt:
        print("\nExiting...")
        terminate_flag.set()
        if proc:
            proc.kill()
        if conn:
            conn.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
