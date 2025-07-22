import pika
import pickle
import math
import cv2
import time

from DetectionAndViolation.violation_database import init_db, save_violation

class HandState:
    def __init__(self, hand_id):
        self.hand_id = hand_id
        self.entered_roi_at_least_once = False
        self.currently_in_roi = False
        self.exited_roi_after_entry = False
        self.touched_pizza_after_roi_exit = False
        self.had_scooper_when_needed = False
        self.violation_recorded = False
        self.last_seen_frame = 0

def is_inside_roi(box, roi_list):
    if not roi_list:
        return False
    x1, y1, w1, h1 = box
    for roi in roi_list:
        x2, y2, w2, h2 = roi
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h1)
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        if inter_area > 0:
            return True
    return False

def get_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def are_boxes_close(box1, box2, threshold=37):
    center1 = get_center(box1)
    center2 = get_center(box2)
    dx = (center1[0] - center2[0])**2
    dy = (center1[1] - center2[1])**2
    distance = math.sqrt(dx + dy)
    return distance < threshold

hand_states = {}
violations_set = set()
ROI_LIST = [(454, 346, 53, 42), (460, 316, 57, 36), (411, 528, 59, 57)] # Hardcoded as in original main.py
PIZZA_AREA = [] # This will be populated per frame

logical_hands = {}
next_logical_id = 0
HAND_MATCHING_THRESHOLD = 50
HAND_GRACE_PERIOD = 30

def process_frame_logic(tracked_hands, tracked_scoopers, current_frame_number, current_frame, channel):
    current_frame_hand_ids = set()
    for hand_id, hand_box in tracked_hands:
        current_frame_hand_ids.add(hand_id)
        if hand_id not in hand_states:
            hand_states[hand_id] = HandState(hand_id)

        state = hand_states[hand_id]
        state.last_seen_frame = current_frame_number

        was_currently_in_roi = state.currently_in_roi
        state.currently_in_roi = is_inside_roi(hand_box, ROI_LIST)

        if state.currently_in_roi:
            state.entered_roi_at_least_once = True
        
        if was_currently_in_roi and not state.currently_in_roi and state.entered_roi_at_least_once:
            state.exited_roi_after_entry = True
            
        if is_inside_roi(hand_box, PIZZA_AREA):
            if state.exited_roi_after_entry:
                state.touched_pizza_after_roi_exit = True
            
        hand_has_scooper_now = False
        for scooper_id, scooper_box in tracked_scoopers:
            if are_boxes_close(hand_box, scooper_box):
                hand_has_scooper_now = True
                break
        state.had_scooper_when_needed = hand_has_scooper_now

        if (
            state.entered_roi_at_least_once and
            state.exited_roi_after_entry and
            state.touched_pizza_after_roi_exit and
            not state.had_scooper_when_needed and
            not state.violation_recorded
        ):
            print(f"A violation of the hand ID was created: {hand_id} In frame No.{current_frame_number}")
            violations_set.add(hand_id)
            state.violation_recorded = True
            save_violation(current_frame, hand_id, hand_box)

    keys_to_delete = [
        hand_id for hand_id, hand_state in hand_states.items()
        if current_frame_number - hand_state.last_seen_frame > 30
    ]
    for key in keys_to_delete:
        if key in hand_states:
            del hand_states[key]

    violation_count = len(violations_set)

    # Draw annotations on the frame
    for hand_id, hand_box in tracked_hands:
        x1, y1, w, h = hand_box
        color = (0, 0, 255) if hand_id in violations_set else (0, 255, 0)
        cv2.rectangle(current_frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        cv2.putText(current_frame, f"hand-{hand_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for roi in ROI_LIST:
        x, y, w, h = roi
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(current_frame, "ROI", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(current_frame, f"Violations: {violation_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Publish annotated frame and violation count to RabbitMQ
    send_data = {
        'frame': current_frame,
        'timestamp': time.time(),
        'number_of_violation': violation_count
    }
    send_body = pickle.dumps(send_data)
    channel.basic_publish(exchange='', routing_key='annotated_frames', body=send_body)

    return violation_count

def update_logical_hands(detected_hands, frame_count):
    global next_logical_id
    stabilized_hands = []
    matched_logical_ids = set()

    for det_id, det_box in detected_hands:
        best_match_id = None
        min_dist = float('inf')

        for log_id, log_data in logical_hands.items():
            if log_id in matched_logical_ids:
                continue
            
            dist = math.sqrt((get_center(det_box)[0] - get_center(log_data['box'])[0])**2 + 
                             (get_center(det_box)[1] - get_center(log_data['box'])[1])**2)

            if dist < HAND_MATCHING_THRESHOLD and dist < min_dist:
                min_dist = dist
                best_match_id = log_id

        if best_match_id is not None:
            logical_hands[best_match_id]['box'] = det_box
            logical_hands[best_match_id]['last_seen'] = frame_count
            stabilized_hands.append((best_match_id, det_box))
            matched_logical_ids.add(best_match_id)
        else:
            logical_hands[next_logical_id] = {'box': det_box, 'last_seen': frame_count}
            stabilized_hands.append((next_logical_id, det_box))
            next_logical_id += 1

    expired_ids = [log_id for log_id, log_data in logical_hands.items() 
                   if frame_count - log_data['last_seen'] > HAND_GRACE_PERIOD]
    for log_id in expired_ids:
        del logical_hands[log_id]

    return stabilized_hands

def callback(ch, method, properties, body):
    global PIZZA_AREA
    data = pickle.loads(body)
    frame_number = data['frame_number']
    detected_objects = data['detected_objects']
    frame = data['frame']

    tracked_hand = []
    tracked_scooper = []
    PIZZA_AREA.clear()

    for obj in detected_objects:
        if obj['label'] == "hand":
            tracked_hand.append((obj['id'], obj['box']))
        elif obj['label'] == "scooper":
            tracked_scooper.append((obj['id'], obj['box']))
        elif obj['label'] == 'pizza':
            x1, y1, w, h = obj['box']
            w_new = int(w * 1.2)
            h_new = int(h * 1.2)
            x1_new = x1 - (w_new - w) // 2
            y1_new = y1 - (h_new - h) // 2
            enlarged_pizza_box = (x1_new, y1_new, w_new, h_new)
            PIZZA_AREA.append(enlarged_pizza_box)

    stabilized_hands = update_logical_hands(tracked_hand, frame_number)
    process_frame_logic(stabilized_hands, tracked_scooper, frame_number, frame, ch)

def main():
    init_db()
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue='detected_objects')
    channel.queue_declare(queue='annotated_frames') # Declare new queue for annotated frames

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.basic_consume(queue='detected_objects', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

if __name__ == '__main__':
    main()
