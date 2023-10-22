from ultralytics import YOLO
import cv2


model= YOLO('../Yolo-weights/best.pt')
results= model("test images/1.jpg", show=True, show_labels=False, show_conf=False)
cv2.waitKey(0)