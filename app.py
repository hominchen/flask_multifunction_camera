from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

# 全局變量
global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

# 製作目錄
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# dnn檢測模型    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', 
    './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

# === def 函式 ===
# rec變量為true時將幀寫入avi
def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

# 在幀中檢測人臉的裁剪幀
def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]

    # 對圖像進行預處理，返回4通道的blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0))
    
    # 將處理過的blob帶入到net()模型中
    net.setInput(blob)
    detections = net.forward()

    confidence = detections[0, 0, 0, 2]

    # 提取與預測相關的置信度(即機率)
    if confidence < 0.5:            
        return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame
 
# 從相機對象捕獲幀
def gen_frames():
    global out, capture,rec_frame
    while True:
        success, frame = camera.read()
        if success:
            # 檢查是否有任何濾鏡開關為真
            if(face):                
                frame= detect_face(frame)
            if grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)          
            # 如果rec為真，則將幀複製到rec_frame全局變量
            if(rec):
                rec_frame=frame
                # cv2.putText(圖片影像/繪製的文字/左上角坐標/字體/字體大小/顏色/字體粗細/字體線條種類)
                frame= cv2.putText(cv2.flip(frame,1),
                    "Recording...",
                    (0,25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
            try:
                # 將幀編碼到內存緩衝區中轉換為字節數組
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                # 作為HTTP 響應發送的格式生成幀數據
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


# === 主程式/路由 ===
@app.route('/')
def index():
    return render_template('index.html')
    
# 設置html文件中的圖像源
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

# 處理所有的開關和視頻錄製
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        # 多個submit type
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey  #先取得全域狀態後，反轉返回
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        
        # 啟動一個新線程將幀錄製到視頻中
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # cv2.VideoWriter(文件的路徑/指定編碼器/視頻的幀率/文件的畫面尺寸/黑白還是彩色的畫面)
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), 
                    fourcc, 20.0, (640, 480))
                # 開始錄製視頻的新線程執行緒，目標record()函式
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     