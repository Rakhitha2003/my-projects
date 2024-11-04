import cv2
import numpy as np
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_vehicles(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    vehicle_count = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: 
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

               
                if class_id in [2, 3, 5, 7]:  
                    vehicle_count += 1

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return vehicle_count, frame

def adjust_green_light(vehicle_count):
   
    base_green_time = 30  
    additional_time = min(vehicle_count * 5, 60) 
    return base_green_time + additional_time


image_folder = "path/to/image/dataset"  
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

total_frames = len(image_files) 

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        continue 

    vehicle_count, output_frame = detect_vehicles(frame)
    green_light_duration = adjust_green_light(vehicle_count)

   
    cv2.putText(output_frame, f'Vehicle Count: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_frame, f'Green Light Duration: {green_light_duration}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
    cv2.imshow("Traffic Management System", output_frame)
    cv2.waitKey(2000)  
    output_image_path = os.path.join(image_folder, f'output_{image_file}')
    cv2.imwrite(output_image_path, output_frame)


print(f'Total frames processed: {total_frames}')

cv2.destroyAllWindows()
