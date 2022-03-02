from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from time import strftime
from threading import Thread
from arduino import *
import socket,cv2, pickle,struct

#Send frame to server
def send_to_server(frame):
    client_socket.sendall(b"sent")
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),100]
    result, image = cv2.imencode('.jpg', frame, encode_param)
    if result:
        data = pickle.dumps(image, 0)
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)
        print("Sent image to server")
    return

#Recived result detection from server
def recv_from_server():
    global msg
    msg = client_socket.recv(8)
    if msg != b"None":
        print('Recived msg from server:', "Bất thường" if msg == b'1' else "Bình thường")
    else:
        print('Recived msg from server:', "Không tìm thấy sản phẩm")
    update_text()
    return

#Stop button function
def stop_button():
    client_socket.sendall(b"stop")
    client_socket.close()
    root.destroy()
    print("Stop client socket and shut down gui")


#Show timestamp function
def show_time():
    string = strftime('%H:%M:%S %p')
    timestamp.config(text = string)
    timestamp.after(1000, show_time)

#Update frame webcam show on canvas
def update_frame():
    global photo, img_counter
    _, frame = cap.read()
    # frame = cv2.imread("test/exp30-000.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    my_canvas.create_text(680,200,text="WEBCAM", font = ('Arial', 32, 'bold'),fill='blue', anchor='nw')
    
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    my_canvas.create_image(450,250, image = photo, anchor="nw") 
    
    if img_counter%100 == 0:
        thread1 = Thread(target=send_to_server, args=(frame,))
        thread2 = Thread(target=recv_from_server)
        thread1.start()
        thread2.start()
    
    img_counter += 1
    root.after(15, update_frame)
   
#Update text show on canvas, take result from server
def update_text():
    if msg != b"None":
        my_canvas.delete("ketqua2")
        my_canvas.create_text(660 if msg ==b'1' else 650,740,text="Bất thường" if msg == b"1" else "Bình thường", font = ('Arial', 32, 'bold'),fill='red' if msg == b"1" else 'blue', anchor='nw', tag='ketqua1')
    else:
        my_canvas.delete("ketqua1")
        my_canvas.create_text(500,750,text="Không tìm thấy sản phẩm", font = ('Arial', 32, 'bold'),fill='red', anchor='nw', tag="ketqua2")

    
#Client socket initialize
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '10.10.56.85'
port = 9999
client_socket.connect((host_ip,port)) 

#Global variable
img_counter = 0
photo = None

#Capture video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Width and Height video show on canvas
canvas_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
canvas_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
   
#Gui initialize        
root = Tk()
root.title("Đồ án tốt nghiệp")
root.geometry("1920x1080")
root.attributes('-fullscreen',True)

#width and height gui
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

#images show on gui
my_img1 = ImageTk.PhotoImage(Image.open("images/smartbuilding.png").resize((w,h)))
my_img2 = ImageTk.PhotoImage(Image.open("images/logo_bk.jpg").resize((150,150)))
my_img3 = ImageTk.PhotoImage(Image.open("images/logo_khoadien.png").resize((150,150)))

#initialize canvas on gui
my_canvas = Canvas(root, width=1920, height=1080) 
my_canvas.pack(fill=BOTH, expand=True)

#create text and image on canvas
my_canvas.create_image(0,0, image = my_img1, anchor='nw')
my_canvas.create_image(0,0, image=my_img2, anchor='nw')
my_canvas.create_image(0,160, image=my_img3, anchor='nw')
my_canvas.create_text(550,20,text="ĐỒ ÁN TỐT NGHIỆP", font = ('Arial', 35, 'bold'), anchor='nw')
my_canvas.create_text(250,80,text="Đề tài: Ứng dụng xử lý ảnh trong phân loại sản phẩm lỗi", font = ('Arial', 32, 'bold'), anchor='nw')

#timestamp function and position
timestamp = Label(my_canvas, font = ('calibri', 30, 'bold'),
            background = 'white',
            foreground = 'black')
timestamp.place(x=1300,y=20,anchor='nw')

#style font button
st = Style()
st.configure('W.TButton', background='#345', foreground='black', font=('calibri', 18, 'bold'))

#Create button
run_button = Button(root, text="RUN", style="W.TButton", command=update_frame)
stop_button = Button(root, text="STOP", style="W.TButton", command=stop_button)

#Position of Button
my_canvas.create_window(1200,800,anchor="nw", window=run_button)
my_canvas.create_window(1350,800,anchor="nw", window=stop_button)

#function showtime call
show_time()

#Show gui
root.mainloop() 