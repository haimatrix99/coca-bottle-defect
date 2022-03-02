import socket, cv2, pickle,struct
from detect import *

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# host_name  = socket.gethostname()
# host_ip = socket.gethostbyname(host_name)
host_ip = '10.10.56.85'
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)


server_socket.bind(socket_address)

server_socket.listen(5)
print("LISTENING AT:",socket_address)

data = b""
payload_size = struct.calcsize(">L")

client_socket,addr = server_socket.accept()

print('GOT CONNECTION FROM:',addr)
    
while True:
    data_recv = client_socket.recv(64)
    if data_recv == b"sent":
        print("Recived image from client")
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data+=packet
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L",packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
            
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        frame = cv2.cvtColor(cv2.imdecode(pickle.loads(frame_data), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        frame, res = detect_image(frame)
        print("Processing image")
        if res is not None:
            msg = "1" if res > od.threshold else "0"
            print("Sent back msg to client:", "Bất thường" if msg == "1" else "Bình thường")
        else:
            msg = "None"
            print("Không tìm thấy sản phẩm trên webcam")
        client_socket.sendall(bytes(msg, 'utf-8'))

    if data_recv == b"stop":
        client_socket.close()
        print("Stop server socket")
        break

server_socket.close()
