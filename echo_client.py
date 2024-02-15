import socket
import tqdm
import os

def client():
    # TCP/IP socket is created
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Spacer to separate the frame from the file
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096 #4KB
    filename = "ocr_result.txt" #보낼파일명
    # We get the size of the file
    filesize = os.path.getsize(filename)
    # The name and port of the server host are indicated
    server_address = ('192.168.137.147', 5001) # change server address
    print('Conexion a: {} puerto: {}'.format(*server_address))
    sock.connect(server_address)
    # Send the file through the socket
    sock.send(f"{filename}{SEPARATOR}{filesize}".encode())
    # Progress bar
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            # Read the number of bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            # It is verified that the information has been sent by the socket
            sock.sendall(bytes_read)
            # Update bar progress
            progress.update(len(bytes_read))
    sock.close()