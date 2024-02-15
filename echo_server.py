import os
import socket
import tqdm

def server():
    # If it creates a TCP / IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # The created socket is bound to local port 10000
    server_address = ('192.168.137.43', 5001) #서버주소 바꾸기
    sock.bind(server_address)
    # Receive 4096 bytes
    BUFFER_SIZE = 4096
    SEPARATOR = "<SEPARATOR>"
    # The server listens for a connection request.
    sock.listen(1)
    # Waiting for the connection as long as I don't change the heat to 0
    print('waiting for client')
    connection, client_address = sock.accept()
    print('Connected address: ', client_address)
    # The information of the incoming file is received through the client socket
    received = connection.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    filename = os.path.basename(filename)
    # Convert file size to integer
    filesize = int(filesize)
    # Progress bar indicating how much information has been obtained
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        while True:
            # Read the value of the buffer that is receiving
            bytes_read = connection.recv(BUFFER_SIZE)
            if not bytes_read:
                break
            # When the budder finishes reading, it writes the data to a file.
            f.write(bytes_read)
            # Update the progress bar
            progress.update(len(bytes_read))
    # The connection is closed when it finishes receiving information
    connection.close()