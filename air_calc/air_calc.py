import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from collections import deque

class AirCalculator:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.points = deque(maxlen=512)
        self.canvas = np.zeros((400, 400), dtype=np.uint8)
        
        # ONNX session for MNIST model
        self.session = ort.InferenceSession("mnist-12.onnx")
        
        # State management
        self.drawing_enabled = False
        self.review_mode = True
        self.current_prediction = None
        self.confidence = None
        self.expression = []

    # Preprocess canvas to match the model input requirements
    def preprocess_canvas(self, canvas):
        # Resize to 28x28
        resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # Reshape to (1, 1, 28, 28) as required by the model
        processed = np.reshape(normalized, (1, 1, 28, 28))

        return processed

    # Apply softmax to logits
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))  # Prevent overflow
        return exp_logits / np.sum(exp_logits)


    # Predict digit from the canvas
    def predict_digit(self):
        if np.sum(self.canvas) == 0:
            return None, None

        # Preprocess the canvas
        processed = self.preprocess_canvas(self.canvas)

        # Predict using ONNX
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: processed})[0]

        # Debugging: Print the shape of the output
        print(f"Model output shape: {output.shape}")
        print(f"Model output: {output}")

        # Apply softmax to get probabilities
        probabilities = self.softmax(output.flatten())  # Ensure output is a 1D array
        print(f"Softmax probabilities: {probabilities}")

        digit = np.argmax(probabilities)
        confidence = probabilities[digit]
        return digit, confidence
    # Count raised fingers based on hand landmarks
    def count_raised_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]
        count = 0
        for tip_id in finger_tips:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                count += 1
        return count

    # Map finger count to operators
    def get_operator_from_gesture(self, finger_count):
        operator_map = {
            2: '+',
            3: '-',
            4: '*',
            5: '/'
        }
        return operator_map.get(finger_count, None)

    # Evaluate the mathematical expression
    def evaluate_expression(self):
        if not self.expression:
            return "0"
        try:
            expr_str = ''.join(str(item) for item in self.expression)
            result = eval(expr_str)
            return str(result)
        except:
            return "Error"

    # Draw status box on the frame
    def draw_status_box(self, frame, text, position, color=(0, 255, 0)):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Main application loop
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and not self.review_mode:
                landmarks = results.multi_hand_landmarks[0]
                
                index_tip = landmarks.landmark[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                canvas_x = int(x * 400 / w)
                canvas_y = int(y * 400 / h)
                
                if self.drawing_enabled:
                    self.points.append((canvas_x, canvas_y))
                    points_list = list(self.points)
                    if len(points_list) > 1:
                        cv2.line(self.canvas, points_list[-2], points_list[-1], 255, 3)
                        cv2.line(frame, (x, y), 
                                 (int(points_list[-2][0] * w / 400), 
                                  int(points_list[-2][1] * h / 400)), 
                                 (0, 0, 255), 2)
            
            canvas_display = cv2.resize(self.canvas, (200, 200))
            frame[50:250, w-250:w-50] = cv2.cvtColor(canvas_display, cv2.COLOR_GRAY2BGR)
            
            if self.review_mode:
                if self.current_prediction is not None:
                    status = f"Predicted: {self.current_prediction} (Confidence: {self.confidence:.2f})"
                    self.draw_status_box(frame, status, (10, 30))
                    self.draw_status_box(frame, "Press 'Y' to accept, 'N' to redraw", (10, 60))
                else:
                    self.draw_status_box(frame, "No prediction available. Press 'N' to redraw", (10, 30))
            else:
                if self.drawing_enabled:
                    self.draw_status_box(frame, "Drawing... Press SPACE to finish", (10, 30))
                else:
                    self.draw_status_box(frame, "Press SPACE to start drawing", (10, 30))
            
            expr_str = ''.join(str(x) for x in self.expression)
            result = self.evaluate_expression()
            self.draw_status_box(frame, f"Expression: {expr_str} = {result}", (10, h-90))
            
            cv2.putText(frame, "2-5 fingers: Operators | G: Gesture mode | C: Clear | Q: Quit", 
                       (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Air Calculator', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and not self.review_mode:
                if not self.drawing_enabled:
                    self.drawing_enabled = True
                    self.canvas = np.zeros((400, 400), dtype=np.uint8)
                    self.points.clear()
                else:
                    self.drawing_enabled = False
                    self.review_mode = True
                    self.current_prediction, self.confidence = self.predict_digit()
            elif key == ord('y') and self.review_mode:
                if self.current_prediction is not None:
                    self.expression.append(self.current_prediction)
                self.review_mode = False
                self.canvas = np.zeros((400, 400), dtype=np.uint8)
                self.points.clear()
            elif key == ord('n') and self.review_mode:
                self.review_mode = False
                self.canvas = np.zeros((400, 400), dtype=np.uint8)
                self.points.clear()
            elif key == ord('g'):
                if results.multi_hand_landmarks:
                    finger_count = self.count_raised_fingers(results.multi_hand_landmarks[0])
                    operator = self.get_operator_from_gesture(finger_count)
                    if operator:
                        self.expression.append(operator)
            elif key == ord('c'):
                self.canvas = np.zeros((400, 400), dtype=np.uint8)
                self.points.clear()
                self.expression = []
                self.review_mode = False
                self.drawing_enabled = False
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calculator = AirCalculator()
    calculator.run()
