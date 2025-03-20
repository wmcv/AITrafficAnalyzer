from collections import defaultdict, deque
from typing import Any

import cv2
from ultralytics import YOLO
import numpy as np

vid_path = 'video.mp4'
#vid_path = 'video.mp4'
model = YOLO('yolov8s.pt')
class_list = ['car', 'motorcycle', 'bus', 'truck']

cap = cv2.VideoCapture(vid_path)
frame_count = 0
mask = cv2.imread('mask.png')

left_resizer_x = 510//11
left_resizer_y = 800//54
right_resizer_x = 510//11
right_resizer_y = 800//72

#variables for left side of the road (heading north)
count_zone_left = [(300,540),(910,830)]
car_count_left = 0
truck_count_left = 0
bus_count_left = 0
counted_left = set()
speed_tot_left = 0
speed_num_left = 0
speed_fallback_left = 0
SOURCE_LEFT = np.array([[690,496],[905,496],[814,812],[175,812]])
TARGET_WIDTH_LEFT = 11
TARGET_HEIGHT_LEFT = 54
TARGET_LEFT = np.array(
    [
        [0,0],
        [TARGET_WIDTH_LEFT-1, 0],
        [TARGET_WIDTH_LEFT-1, TARGET_HEIGHT_LEFT-1],
        [0, TARGET_HEIGHT_LEFT-1]
    ]
)

#variables for right side of the road (heading south)
count_zone_right = [(1000,540),(1610,830)]
car_count_right = 0
truck_count_right = 0
bus_count_right = 0
counted_right = set()
speed_tot_right = 0
speed_num_right = 0
speed_fallback_right = 0
SOURCE_RIGHT = np.array([[1000,474],[1180,474],[1726,830],[1107,832]])
TARGET_WIDTH_RIGHT = 11
TARGET_HEIGHT_RIGHT = 72
TARGET_RIGHT = np.array(
    [
        [0,0],
        [TARGET_WIDTH_RIGHT-1, 0],
        [TARGET_WIDTH_RIGHT-1, TARGET_HEIGHT_RIGHT-1],
        [0, TARGET_HEIGHT_RIGHT-1]
    ]
)

#function to calculate the critical distance for a car to stop
def crit_distance_calc(speed, road_condition="dry"):
    reaction_time = 1.0
    friction_coef = {
        "dry": 0.85,
        "wet": 0.5,
        "icy": 0.2
    }
    mu = friction_coef.get(road_condition, 0.7)
    d_reaction = speed * reaction_time
    d_braking = (speed ** 2) / (2 * mu * 9.81)
    return d_reaction + d_braking


#class to transform road perspective and points into birds eye view coord system
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1,1,2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2)


view_transformer_left = ViewTransformer(source=SOURCE_LEFT, target=TARGET_LEFT)
view_transformer_right = ViewTransformer(source=SOURCE_RIGHT, target=TARGET_RIGHT)
coordinates = defaultdict(lambda: deque(maxlen=16))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    #frame to visualize non-affine transformation used in speed calculation
    speed_perspective_frame = np.zeros((800, 920, 3), dtype=np.uint8)
    cv2.rectangle(speed_perspective_frame, (0,0), (920, 800), (30,30,30), -1)
    cv2.rectangle(speed_perspective_frame, (510 - 70, 0), (510 - 20, 800), (125, 125, 125), -1)
    cv2.line(speed_perspective_frame, (130, 0), (130, 800), (0, 255, 255), 3)
    cv2.line(speed_perspective_frame, (290, 0), (290, 800), (0, 255, 255), 3)
    cv2.line(speed_perspective_frame, (640, 0), (640, 800), (0, 255, 255), 3)
    cv2.line(speed_perspective_frame, (790, 0), (790, 800), (0, 255, 255), 3)

    #frame skip for faster performance and reset speed averages
    frame_count += 1
    if frame_count % 2 != 0:
        speed_tot_left = 0
        speed_num_left = 0
        speed_tot_right = 0
        speed_num_right = 0
        continue

    #mask region for optimization
    frameRegion = cv2.bitwise_and(frame, mask)

    results = model.track(frameRegion, conf=0.3, iou=0.5, persist=True, tracker="bytetrack.yaml")

    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    car_lst = []

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = box.conf[0]
            tracking_id = int(box.id[0]) if box.id is not None else -1
            class_name = model.names[cls]
            midpt_x, midpt_y = (x1+x2)//2, y2

            #adds tally to number of cars on the left side
            if count_zone_left[0][0] < midpt_x < count_zone_left[1][0] and count_zone_left[0][1] < midpt_y < count_zone_left[1][1]:
                if tracking_id not in counted_left:
                    if class_name in ['car', 'motorcycle']:
                        car_count_left+=1
                    elif class_name == 'truck':
                        truck_count_left+=1
                    elif class_name == 'bus':
                        bus_count_left+=1
                    counted_left.add(tracking_id)

            #adds tally to number of cars on the right side
            elif count_zone_right[0][0] < midpt_x < count_zone_right[1][0] and count_zone_right[0][1] < midpt_y < count_zone_right[1][1]:
                if tracking_id not in counted_right:
                    if class_name in ['car', 'motorcycle']:
                        car_count_right += 1
                    elif class_name == 'truck':
                        truck_count_right += 1
                    elif class_name == 'bus':
                        bus_count_right += 1
                    counted_right.add(tracking_id)

            bounding_pts = np.array([[midpt_x, midpt_y]], dtype=np.float32)

            #use appropriate coord system based on which side of the road the car is on
            if midpt_x < 962:
                bounding_pts = view_transformer_left.transform_points(points=bounding_pts).astype(int)
            else:
                bounding_pts = view_transformer_right.transform_points(points=bounding_pts).astype(int)

            #track car's position in a dict
            x, y = bounding_pts[0]
            coordinates[tracking_id].append(y)
            speed = -1

            #if the coord system position tracking is half full, start calculating speed
            if len(coordinates[tracking_id]) < 8:
                text = str(0)
            else:
                coordinate_start = coordinates[tracking_id][-1]
                coordinate_end = coordinates[tracking_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracking_id]) / 16
                speed = (distance / time) * 3.6
                text = f'{speed:.1f} km/h'

            #change speeding cards bbox to red
            speed_color = (0,255,0)
            if speed > 113:
                speed_color = (0,0,255)

            if midpt_x < 962:
                x = int(x * left_resizer_x)
            else:
                x = int(x * right_resizer_x) + 510

            if speed != -1:
                crit_dist = 0.278 * speed + ((speed*speed)/(254*0.7))
                car_lst.append([speed, crit_dist, [x, y]])

            #visualize the cars on the non-affine transformation frame
            if midpt_x < 962:
                cv2.rectangle(speed_perspective_frame, (x - 20, int(y * left_resizer_y) - 30), (x + 20, int(y * left_resizer_y) + 30), speed_color, -1)
                cv2.putText(speed_perspective_frame, f'{speed:.1f} km/h', (x - 40, int(y * left_resizer_y) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                cv2.rectangle(speed_perspective_frame, (x - 20, int(y * right_resizer_y) - 30), (x + 20, int(y * right_resizer_y) + 30), speed_color, -1)
                cv2.putText(speed_perspective_frame, f'{speed:.1f} km/h', (x - 40, int(y * right_resizer_y) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            #add the speed to its respective side's total speed
            if speed != -1:
                if midpt_x < 962:
                    speed_tot_left += speed
                    speed_num_left += 1
                    speed_fallback_left = speed
                else:
                    speed_tot_right += speed
                    speed_num_right += 1
                    speed_fallback_right = speed

            text = f'{class_name} | {text}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            if conf > 0.3 and class_name in ['car', 'motorcycle', 'bus', 'truck']:
                cv2.rectangle(frame, (x1,y1), (x2,y2), speed_color, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 200, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    #width limits per lane
    lane_dict = {
        0: [0, 130],
        1: [130, 290],
        2: [290, 440],
        3: [490, 640],
        4: [640, 790],
        5: [790, 920]
    }

    car_lst_left = sorted(car_lst, key=lambda car: car[2][1])  # low to high
    car_lst_right = sorted(car_lst, key=lambda car: car[2][1], reverse=True)

    #function to calculate the stopping-time risk for cars
    def SpeedDistanceRisk(num_cars, car_lst, lane_num=0):
        risk = 0
        lane = lane_dict[lane_num]
        for i in range(num_cars):
            for j in range(i + 1, num_cars):
                if not ((lane[0] < car_lst[i][2][0] < lane[1]) and (lane[0] < car_lst[j][2][0] < lane[1])):
                    continue

                if car_lst[i][0] >= car_lst[j][0]:
                    continue
                distance = np.linalg.norm(np.array(car_lst[i][2][1]) - np.array(car_lst[j][2][1]))
                dist_crit = crit_distance_calc(max(car_lst[i][0], car_lst[j][0]), road_condition="dry")
                if distance < dist_crit + 3:
                    risk += (dist_crit - distance) / dist_crit
        return risk

    #calculate stopping-time risk per car in each lane
    risk_0 = SpeedDistanceRisk(speed_num_left, car_lst_left, 0)
    risk_1 = SpeedDistanceRisk(speed_num_left, car_lst_left, 1)
    risk_2 = SpeedDistanceRisk(speed_num_left, car_lst_left, 2)
    risk_3 = SpeedDistanceRisk(speed_num_right, car_lst_right, 3)
    risk_4 = SpeedDistanceRisk(speed_num_right, car_lst_right, 4)
    risk_5 = SpeedDistanceRisk(speed_num_right, car_lst_right, 5)

    #function to calculate car count for a given lane
    def lane_counter(car_lst, lane=0):
        count = 0
        lane = lane_dict[lane]
        for car in car_lst:
            if lane[0] < car[2][0] < lane[1]:
                count+=1
        return count

    #calculate cars in each lane
    lane_count_0 = lane_counter(car_lst, 0)
    lane_count_1 = lane_counter(car_lst, 1)
    lane_count_2 = lane_counter(car_lst, 2)
    lane_count_3 = lane_counter(car_lst, 3)
    lane_count_4 = lane_counter(car_lst, 4)
    lane_count_5 = lane_counter(car_lst, 5)

    #calculate the car density for each lane
    car_density_0 = lane_count_0 / TARGET_HEIGHT_LEFT
    car_density_1 = lane_count_1 / TARGET_HEIGHT_LEFT
    car_density_2 = lane_count_2 / TARGET_HEIGHT_LEFT
    car_density_3 = lane_count_3 / TARGET_HEIGHT_RIGHT
    car_density_4 = lane_count_4 / TARGET_HEIGHT_RIGHT
    car_density_5 = lane_count_5 / TARGET_HEIGHT_RIGHT

    alpha_density = 0.5
    alpha_speeddistance = 0.65

    #calculate risk on left side
    car_density_left = (car_density_0 + car_density_1 + car_density_2)/3
    speeddistance_left = (risk_0 + risk_1 + risk_2)/3
    scaled_density_left = car_density_left * 10
    normalized_density_left = scaled_density_left
    final_risk_left = (speeddistance_left*alpha_speeddistance + normalized_density_left*alpha_density)*100

    #calculate risk on right side
    car_density_right = (car_density_3 + car_density_4 + car_density_5)/3
    speeddistance_right = (risk_3 + risk_4 + risk_5)/3
    scaled_density_right = car_density_right * 10
    normalized_density_right = scaled_density_right
    final_risk_right = (speeddistance_right * alpha_speeddistance + normalized_density_right * alpha_density)*100

    if final_risk_left > 99.9:
        final_risk_left = 99.9
    if final_risk_left > 99.9:
        final_risk_left = 99.9

    cv2.rectangle(frame, (135,25), (915,295), (255, 200, 0), -1)
    cv2.rectangle(frame, (1035, 25), (1835, 295), (255, 200, 0), -1)

    #left side (car counter)
    cv2.rectangle(frame, count_zone_left[0], count_zone_left[1], (230, 230, 230), 2)
    car_text_left = f'Cars Southbound: {car_count_left}'
    truck_text_left = f'Trucks Southbound: {truck_count_left}'
    bus_text_left = f'Buses Southbound: {bus_count_left}'
    if speed_num_left:
        speed_text_left = f'Average Speed Southbound: {(speed_tot_left/speed_num_left):.1f} km/h'
    else:
        speed_text_left = f'Average Speed Southbound: {speed_fallback_left} km/h'
    risk_text_left = f'Accident Risk Chance: {final_risk_left:.1f}%'
    cv2.putText(frame, car_text_left, (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, truck_text_left, (150, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, bus_text_left, (150, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, speed_text_left, (150, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, risk_text_left, (150, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)

    #right side (car counter)
    text_right = f'Cars Northbound: {car_count_right}'
    truck_text_right = f'Trucks Northbound: {truck_count_right}'
    bus_text_right = f'Buses Northbound: {bus_count_right}'
    if speed_num_right:
        speed_text_right = f'Average Speed Northbound: {(speed_tot_right/speed_num_right):.1f} km/h'
    else:
        speed_text_right = f'Average Speed Northbound: {speed_fallback_right} km/h'
    risk_text_right = f'Accident Risk Chance: {final_risk_right:.1f}%'
    cv2.rectangle(frame, count_zone_right[0], count_zone_right[1], (230, 230, 230), 2)
    cv2.putText(frame, text_right, (1050, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, truck_text_right, (1050, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, bus_text_right, (1050, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, speed_text_right, (1050, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, risk_text_right, (1050, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)

    cv2.imshow("Non-Affine Transformation Visualization", speed_perspective_frame)
    cv2.imshow('AI Car Counting', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()