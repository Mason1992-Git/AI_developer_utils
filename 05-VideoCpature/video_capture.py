import cv2
import os

path = r"F:\risking_driving_0722\cls"

for cls in os.listdir(path):
    save_path = os.path.join(path,cls,"save")
    for file_name in os.listdir(os.path.join(path,cls)):

        if file_name[-4:] != ".mp4":
            print(f"{file_name} is not video file...")
            break
        if not os.path.exists(os.path.join(save_path,file_name)):
            os.makedirs(os.path.join(save_path,file_name))

        file_name_path = os.path.join(path,cls,file_name)
        cap = cv2.VideoCapture(file_name_path)
        if not cap.isOpened():
            print("can't' find camera,please check the device!")
            exit()
        fps = cap.get(5)
        print("this video's FPS is:", fps)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 视频的宽度
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 视频的高度
        print("frame_width:", frame_width)
        print("frame_heigth:", frame_height)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # save_video = cv2.VideoWriter("out.avi", fourcc, 300., (640, 480))
        i = 0
        j = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame......")
                break
            # save_video.write(frame)
            #翻转180度
            # (h,w) = frame.shape[:2]
            # center = (w//2,h//2)
            # M = cv2.getRotationMatrix2D(center,180,1.0)
            # frame = cv2.warpAffine(frame, M, (w, h))
            cv2.imshow('frame', frame)

            if j % 15 == 0:
                cv2.imwrite(os.path.join(save_path,file_name,f"{i}.jpg"),frame)
                i += 1
            j += 1

            if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # 释放视频资源
        # save_video.release()
        cv2.destroyWindow("frame")  # 关闭所有窗口








