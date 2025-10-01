import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import cv2  # Aggiunto per visualizzare l'immagine
import time  # Aggiunto per limitare la frequenza di visualizzazione
import threading

rospy.init_node('image_processing', anonymous=True)

bridge = CvBridge()
latest_img = None
img_lock = threading.Lock()
running = True

def image_callback(msg_img):
    global latest_img
    try:
        cv_img = bridge.imgmsg_to_cv2(msg_img, desired_encoding="passthrough")
        with img_lock:
            latest_img = cv_img
    except Exception as e:
        rospy.logerr(f"Errore nel convertire l'immagine: {e}")

def display_thread():
    global running, latest_img
    rate = rospy.Rate(30)  # Limitiamo a 30 fps
    
    while running and not rospy.is_shutdown():
        with img_lock:
            if latest_img is not None:
                img_to_show = latest_img.copy()
                img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)  # Converti in RGB per OpenCV
                timestamp = int(time.time() * 1000)  # Usa il timestamp in millisecondi per un nome unico
                cv2.imwrite(f"frame_{timestamp}.png", img_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            rospy.signal_shutdown("Stream interrotto dall'utente.")
            break
            
        rate.sleep()

def listener():
    global running
    # Creiamo un thread separato per la visualizzazione
    display = threading.Thread(target=display_thread)
    display.daemon = True
    display.start()
    
    # Sottoscrivi al topic delle immagini
    sub = rospy.Subscriber("/xtion/rgb/image_rect_color", Image, image_callback, queue_size=1)
    camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
    print("Camera Info: ", camera_info)
    print(type(camera_info))
    
    try:
        while running and not rospy.is_shutdown():
            rospy.sleep(0.1)  # Controlla periodicamente se dobbiamo terminare
    except KeyboardInterrupt:
        running = False
    
    # Pulizia
    cv2.destroyAllWindows()
    sub.unregister()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
    finally:
        running = False
        cv2.destroyAllWindows()