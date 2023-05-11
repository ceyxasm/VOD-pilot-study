from imageai.Detection import VideoObjectDetection
import os, sys

file=sys.argv[1]
print(file)
execution_path = os.getcwd()

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")


video_detector = VideoObjectDetection()
video_detector.setModelTypeAsRetinaNet()
video_detector.setModelPath(os.path.join(execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
video_detector.loadModel()


video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, f"../../Dataset/custom-imageai/{file}"), output_file_path=os.path.join(execution_path, f"../../DetectedVideos/retinanet/Detected_retina_{file}") ,  frames_per_second=20, per_frame_function=forFrame,  minimum_percentage_probability=30)
