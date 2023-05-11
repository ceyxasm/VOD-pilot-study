from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()


video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                                                                          output_file_path=os.path.join(execution_path, "camera_detected_video"),
                                                                                          frames_per_second=10,
                                                                                          log_progress=True,
                                                                                          minimum_percentage_probability=40,
                                                                                          detection_timeout=1)
