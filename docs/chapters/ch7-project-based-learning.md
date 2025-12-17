---
sidebar_label: 'Chapter 7: Project-Based Learning'
sidebar_position: 8
---

# Chapter 7: Project-Based Learning

## Introduction to Project-Based Learning in Robotics

Project-based learning is one of the most effective approaches to mastering robotics and AI. By working through complete projects from conception to implementation, you develop practical skills, encounter real-world challenges, and learn to integrate different concepts and technologies. This chapter presents several comprehensive projects that build upon the concepts covered in previous chapters.

Each project in this chapter increases in complexity and incorporates more of the concepts learned throughout the book. We'll cover essential robotics projects as well as more advanced applications, providing you with practical experience in implementing physical AI systems.

## Project 1: Line-Following Robot

A line-following robot is an excellent starting project that combines basic electronics, control systems, and simple AI concepts. This project teaches fundamental concepts of sensor feedback, control loops, and navigation.

### Project Overview

A line-following robot uses optical sensors to detect a line on the ground and follows it. The robot typically uses multiple sensors arranged in a line to detect the position of the line relative to the robot, then adjusts its direction to stay on course.

### Required Components

- **Microcontroller**: Arduino Uno, Raspberry Pi, or ESP32
- **Motor Driver**: L298N or similar H-bridge motor driver
- **Motors**: Two DC motors with wheels
- **Wheels**: Two driven wheels plus one castor wheel for stability
- **Optical Sensors**: 3-5 IR sensors or a QRE1113 array
- **Chassis**: Robot frame to mount components
- **Power Supply**: Battery pack appropriate for your motors
- **Wheels and Mounting Hardware**: For robot mobility

### Hardware Assembly

1. **Chassis Construction**
   - Mount the microcontroller, motor driver, and battery securely on the chassis
   - Install the two drive wheels on the motors
   - Add a castor wheel at the front or back for stability

2. **Motor Wiring**
   - Connect motors to the motor driver
   - Connect motor driver power and control pins to the microcontroller
   - Ensure proper power supply for motors

3. **Sensor Installation**
   - Mount optical sensors in a line at the front/bottom of the robot
   - Position sensors to detect the line at different lateral positions
   - Ensure sensors are at an appropriate distance from the ground

### Software Implementation

```cpp
// Arduino code for line-following robot
const int numSensors = 5;
const int sensorPins[] = {A0, A1, A2, A3, A4};
const int motorLeftFwd = 5;
const int motorLeftBck = 6;
const int motorRightFwd = 9;
const int motorRightBck = 10;

int sensorValues[numSensors];
int lastError = 0;
int integral = 0;

void setup() {
  Serial.begin(9600);

  // Set motor control pins as outputs
  pinMode(motorLeftFwd, OUTPUT);
  pinMode(motorLeftBck, OUTPUT);
  pinMode(motorRightFwd, OUTPUT);
  pinMode(motorRightBck, OUTPUT);

  // Calibrate sensors (read sensor values on both white and black surfaces)
  calibrateSensors();
}

void loop() {
  // Read all sensors
  readSensors();

  // Get position of line (0 = far left, 1000 = far right)
  int position = getLinePosition();

  // Calculate PID error (0 = line centered)
  int error = position - 500;  // 500 is centered

  // PID calculations
  integral += error;
  int derivative = error - lastError;

  // PID constants (tune these for your robot)
  float Kp = 0.5;
  float Ki = 0.05;
  float Kd = 1.0;

  float turn = Kp * error + Ki * integral + Kd * derivative;

  // Calculate motor speeds
  int baseSpeed = 150;
  int leftSpeed = baseSpeed + turn;
  int rightSpeed = baseSpeed - turn;

  // Constrain speeds to valid range
  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  // Apply motor speeds
  setMotorSpeeds(leftSpeed, rightSpeed);

  lastError = error;

  delay(20);  // Small delay for stability
}

void readSensors() {
  for (int i = 0; i < numSensors; i++) {
    sensorValues[i] = analogRead(sensorPins[i]);

    // Convert to 0 or 1 based on threshold
    // (0 = white, 1 = black line)
    sensorValues[i] = sensorValues[i] < 512 ? 1 : 0;
  }
}

int getLinePosition() {
  // Calculate weighted average of sensor positions
  long position = 0;
  long total = 0;

  for (int i = 0; i < numSensors; i++) {
    if (sensorValues[i] == 1) {
      position += i * 1000;  // Weight by position
      total += 1000;
    }
  }

  if (total == 0) {
    // No line detected - return last known position
    return 0;  // Or 1000 if it was to the right
  }

  return position / total;
}

void setMotorSpeeds(int left, int right) {
  if (left > 0) {
    analogWrite(motorLeftFwd, left);
    analogWrite(motorLeftBck, 0);
  } else {
    analogWrite(motorLeftFwd, 0);
    analogWrite(motorLeftBck, -left);
  }

  if (right > 0) {
    analogWrite(motorRightFwd, right);
    analogWrite(motorRightBck, 0);
  } else {
    analogWrite(motorRightFwd, 0);
    analogWrite(motorRightBck, -right);
  }
}

void calibrateSensors() {
  // Calibration process (move robot over both white and black surfaces)
  // This is a simplified version - implement longer calibration process in practice
  delay(1000);

  // Read sensor values on white surface
  for (int i = 0; i < numSensors; i++) {
    sensorValues[i] = analogRead(sensorPins[i]);
  }

  delay(1000);

  // Read sensor values on black surface
  for (int i = 0; i < numSensors; i++) {
    int blackValue = analogRead(sensorPins[i]);
    // Calculate threshold as midpoint
    sensorValues[i] = (sensorValues[i] + blackValue) / 2;
  }
}
```

### Tuning and Optimization

The performance of your line-following robot depends heavily on properly tuning the PID (Proportional-Integral-Derivative) controller constants:

- **Kp (Proportional)**: Controls response to current error. Higher values make the robot react faster but can cause oscillation.
- **Ki (Integral)**: Controls response to accumulated error. Helps eliminate steady-state error but can cause instability.
- **Kd (Derivative)**: Controls response to rate of change of error. Dampens oscillations but can amplify noise.

Start with small values and adjust incrementally. Test on a simple track and gradually increase complexity as performance improves.

![](../../static/img/diagrams/line-following-robot.png)

Figure 7.1: Line-following robot with optical sensors arranged to detect the path.

## Project 2: Object-Detection Robot

Building on the computer vision concepts from Chapter 5, this project creates a robot that can detect, identify, and interact with objects in its environment using cameras and AI models.

### Project Overview

The object-detection robot combines mobile robotics with real-time computer vision to identify objects in its environment and perform actions based on those detections. This project demonstrates how AI can be integrated with physical systems to create intelligent behavior.

### Required Components

- **AI Platform**: Raspberry Pi 4 or NVIDIA Jetson Nano
- **Camera**: Pi Camera Module, USB webcam, or CSI camera
- **Mobile Robot Platform**: Either build or use a pre-built platform with motors
- **Gripper Mechanism**: Simple servo-based gripper or more complex robotic arm
- **Power Supply**: Batteries for extended operation
- **Computer**: For model training and deployment

### Software Implementation

```python
import cv2
import numpy as np
import jetson.inference
import jetson.utils
import time

class ObjectDetectionRobot:
    def __init__(self):
        # Load the object detection model
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

        # Initialize camera
        self.camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
        self.display = jetson.utils.glDisplay()

        # Robot state variables
        self.tracking_object = None
        self.target_locked = False

    def detect_and_track(self):
        """Main detection and tracking loop"""
        while self.display.IsOpen():
            # Capture image
            img, width, height = self.camera.CaptureRGBA()

            # Detect objects
            detections = self.net.Detect(img, width, height)

            # Process detections
            if len(detections) > 0:
                # Find the largest detected object of interest
                target_obj = self.select_target_object(detections)
                if target_obj:
                    self.process_target(img, width, height, target_obj)

            # Render the image
            self.display.RenderOnce(img, width, height)

            # Display FPS
            self.display.Title(f"Object Detection | {self.net.GetNetworkFPS():.1f} FPS")

    def select_target_object(self, detections):
        """Select which object to track based on criteria"""
        # For this example, track the closest object
        best_obj = None
        best_score = -1

        for detection in detections:
            # Could add more complex logic here
            # For example, filter by object class, distance, or size

            if detection.ClassID == 1:  # 1 is typically 'person' in COCO dataset
                if detection.Area > best_score:
                    best_obj = detection
                    best_score = detection.Area

        return best_obj

    def process_target(self, img, width, height, target_obj):
        """Process the selected target object"""
        # Calculate position error
        center_x = target_obj.Center[0]
        center_y = target_obj.Center[1]

        # Target is at the center of the frame
        target_center_x = width / 2
        error_x = center_x - target_center_x

        # Determine if target is centered enough to approach
        tolerance = 50  # pixels
        if abs(error_x) < tolerance:
            self.target_locked = True
            self.move_towards_target(target_obj)
        else:
            self.target_locked = False
            self.rotate_to_target(error_x)

        # Draw target box and center line
        self.draw_target_info(img, target_obj, width)

    def draw_target_info(self, img, target, width):
        """Draw visual indicators for target tracking"""
        # Draw center line
        jetson.utils.Overlay.AddLine(img, width//2, 0, width//2, img.height, 255, 0, 0, 255)

        # Draw target bounding box
        left = int(target.Left)
        top = int(target.Top)
        right = int(target.Right)
        bottom = int(target.Bottom)

        jetson.utils.Overlay.AddRect(img, left, top, right, bottom, 0, 255, 0, 200)

        # Draw center of target
        center_x = int(target.Center[0])
        center_y = int(target.Center[1])
        jetson.utils.Overlay.AddCircle(img, center_x, center_y, 10, 0, 255, 0, 200)

    def rotate_to_target(self, error_x):
        """Rotate robot to align with target"""
        turn_speed = min(abs(error_x) / 100.0, 0.5)  # Cap at 0.5 for safety

        if error_x > 0:
            # Turn right
            self.set_motor_speeds(0.3, -0.3)
        else:
            # Turn left
            self.set_motor_speeds(-0.3, 0.3)

        print(f"Rotating to target, error: {error_x:.2f}")

    def move_towards_target(self, target):
        """Move robot towards target object"""
        # Use target size as distance proxy (larger = closer)
        target_size = target.Area
        desired_size = 10000  # Adjust based on your robot and objects
        distance_error = desired_size - target_size

        if abs(distance_error) < 500:  # Target is close enough
            self.set_motor_speeds(0, 0)  # Stop
            print("Target reached!")

            # Perform action (e.g., pick up object)
            self.perform_action()
        else:
            # Move forward/backward based on distance
            speed = min(abs(distance_error) / 2000.0, 0.5)
            if distance_error > 0:
                # Too far - move forward
                self.set_motor_speeds(speed, speed)
            else:
                # Too close - move backward
                self.set_motor_speeds(-speed, -speed)

            print(f"Moving to target, distance error: {distance_error:.2f}")

    def set_motor_speeds(self, left_speed, right_speed):
        """Set robot motor speeds (implement based on your robot platform)"""
        # This is a placeholder - implement based on your specific robot
        # For example, you might send commands via ROS, GPIO, or serial
        print(f"Setting motor speeds: L={left_speed:.2f}, R={right_speed:.2f}")

    def perform_action(self):
        """Perform action when target is reached"""
        # This could be picking up an object, taking a photo, etc.
        print("Performing action at target location")

# Running the object detection robot
if __name__ == "__main__":
    robot = ObjectDetectionRobot()
    robot.detect_and_track()
```

### Advanced Features

Consider adding these features to extend the basic object-detection robot:

**Path Planning Integration**
- Combine object detection with navigation to approach objects in cluttered environments
- Implement obstacle avoidance while tracking objects

**Multiple Object Tracking**
- Track multiple objects over time using association algorithms
- Prioritize targets based on learned criteria

**Gripper Control**
- Add robotic arm or gripper to interact with detected objects
- Implement grasping strategies based on object properties

![](../../static/img/diagrams/object-detection-robot.png)

Figure 7.2: Object-detection robot using camera and AI model to identify and interact with objects.

## Project 3: Voice-Controlled Robot

This project demonstrates how to integrate speech recognition and natural language processing with robotics to create a robot that responds to voice commands.

### Project Overview

The voice-controlled robot uses speech recognition to interpret user commands and actuates accordingly. This project combines audio processing, natural language understanding, and robot control in a unified system.

### Required Components

- **Microcontroller/Single-board computer**: Raspberry Pi 4 or similar
- **Microphone**: USB microphone or dedicated audio board
- **Speakers**: For audio feedback (optional)
- **Mobile Robot Platform**: With motors and wheels
- **Power Supply**: Batteries for portability

### Software Implementation

```python
import speech_recognition as sr
import pyttsx3
import time
import threading
import queue

class VoiceControlledRobot:
    def __init__(self):
        # Initialize speech recognition
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech

        # Robot state
        self.robot_moving = False
        self.command_queue = queue.Queue()

        # Calibrate microphone for ambient noise
        print("Calibrating microphone...")
        with self.mic as source:
            self.r.adjust_for_ambient_noise(source)
        print("Calibration complete!")

        # Robot movement mapping
        self.command_map = {
            'forward': ('move_forward', 2.0),
            'backward': ('move_backward', 2.0),
            'go forward': ('move_forward', 2.0),
            'go backward': ('move_backward', 2.0),
            'turn left': ('turn_left', 1.0),
            'turn right': ('turn_right', 1.0),
            'spin left': ('turn_left', 2.0),
            'spin right': ('turn_right', 2.0),
            'stop': ('stop', 0),
            'halt': ('stop', 0),
            'dance': ('dance', 3.0),
            'explore': ('explore', 10.0),
        }

    def listen_for_commands(self):
        """Continuously listen for voice commands"""
        with self.mic as source:
            while True:
                try:
                    print("Listening for commands...")
                    audio = self.r.listen(source, timeout=1.0, phrase_time_limit=5.0)

                    # Recognize speech
                    command = self.r.recognize_google(audio).lower()
                    print(f"Heard: {command}")

                    # Process command
                    self.process_command(command)

                except sr.WaitTimeoutError:
                    # No command detected, continue listening
                    pass
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    self.speak("Sorry, I didn't catch that")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    self.speak("Sorry, I'm having trouble with speech recognition")

    def process_command(self, command):
        """Process the recognized voice command"""
        # Check for mapped commands
        for key in self.command_map:
            if key in command:
                action, duration = self.command_map[key]
                self.command_queue.put((action, duration))
                self.speak(f"Okay, I will {key}")
                return

        # Handle complex commands
        if 'find' in command and 'object' in command:
            self.speak("I will search for objects")
            self.command_queue.put(('explore', 10.0))
        elif 'come here' in command:
            self.speak("Coming to you")
            self.command_queue.put(('move_to_user', 5.0))
        else:
            self.speak(f"I don't know how to {command}")

    def speak(self, text):
        """Say something using text-to-speech"""
        print(f"Robot says: {text}")
        # In a real implementation, this would speak the text
        # self.tts_engine.say(text)
        # self.tts_engine.runAndWait()

    def execute_robot_commands(self):
        """Execute commands from the queue"""
        while True:
            if not self.command_queue.empty():
                action, duration = self.command_queue.get()
                print(f"Executing {action} for {duration} seconds")

                # Execute the action
                self.execute_action(action, duration)

            time.sleep(0.1)  # Small delay to prevent busy waiting

    def execute_action(self, action, duration):
        """Execute a specific robot action"""
        if action == 'move_forward':
            self.move_robot(0.5, 0.5, duration)  # linear_vel, angular_vel, time
        elif action == 'move_backward':
            self.move_robot(-0.5, 0.0, duration)
        elif action == 'turn_left':
            self.move_robot(0.0, 0.5, duration)
        elif action == 'turn_right':
            self.move_robot(0.0, -0.5, duration)
        elif action == 'stop':
            self.move_robot(0.0, 0.0, 0.1)
        elif action == 'dance':
            self.perform_dance(duration)
        elif action == 'explore':
            self.explore_environment(duration)

    def move_robot(self, linear_vel, angular_vel, duration):
        """Move the robot with specified velocities for duration"""
        # In a real robot, this would control the actual motors
        # For simulation, just print the action
        print(f"Moving: linear={linear_vel}, angular={angular_vel} for {duration}s")

        start_time = time.time()
        while time.time() - start_time < duration:
            # Send motor commands in real implementation
            # This is where you'd interface with your robot's motor control system
            time.sleep(0.1)

        # Stop after duration
        print("Stopping robot")

    def perform_dance(self, duration):
        """Perform a simple dance routine"""
        start_time = time.time()
        step_time = 0.5

        while time.time() - start_time < duration:
            # Simple dance pattern
            self.move_robot(0.0, 1.0, step_time)  # Turn right
            self.move_robot(0.0, -1.0, step_time)  # Turn left
            time.sleep(0.1)

    def explore_environment(self, duration):
        """Simple exploration algorithm"""
        start_time = time.time()

        while time.time() - start_time < duration:
            # In a real implementation, you'd check for obstacles
            # For now, just move forward for a bit then turn
            self.move_robot(0.3, 0.0, 2.0)  # Move forward
            self.move_robot(0.0, 0.5, 1.0)  # Turn right
            time.sleep(0.5)

    def start(self):
        """Start the voice-controlled robot system"""
        # Start listening thread
        listen_thread = threading.Thread(target=self.listen_for_commands)
        listen_thread.daemon = True
        listen_thread.start()

        # Start command execution thread
        command_thread = threading.Thread(target=self.execute_robot_commands)
        command_thread.daemon = True
        command_thread.start()

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down voice-controlled robot")

# To use this class, uncomment the following lines:
# robot = VoiceControlledRobot()
# robot.start()
```

### Enhancing Voice Control

To make your voice-controlled robot more sophisticated, consider these improvements:

**Natural Language Processing**
- Use more sophisticated NLP to understand complex commands
- Implement intent recognition and entity extraction
- Add context awareness

**Conversation Management**
- Implement a dialogue manager for multi-turn conversations
- Add context memory for follow-up commands
- Handle clarifications when commands are ambiguous

**Multimodal Interaction**
- Combine voice with visual feedback
- Use gestures or touch as additional input modalities
- Provide visual confirmation of understood commands

![](../../static/img/diagrams/voice-controlled-robot.png)

Figure 7.3: Voice-controlled robot with microphone for speech recognition and speakers for audio feedback.

## Project 4: Autonomous Navigation Robot

This project integrates multiple concepts from previous chapters to create a robot capable of autonomously navigating to specified locations while avoiding obstacles.

### Project Overview

The autonomous navigation robot combines SLAM (Simultaneous Localization and Mapping), path planning, and obstacle avoidance to navigate unknown environments. This is a complex project that demonstrates many advanced robotics concepts.

### Required Components

- **Computing Platform**: Raspberry Pi 4, NVIDIA Jetson, or similar
- **Sensors**: LIDAR, camera, IMU, wheel encoders
- **Actuators**: Drive motors with encoders
- **Robot Platform**: Differential drive robot chassis
- **Power Supply**: Adequate for sensors and computing platform

### Software Implementation with ROS 2

```python
# This example shows the key concepts of a ROS 2 navigation system
# Full implementation would require a proper ROS 2 environment

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
import math

class AutonomousNavigator(Node):
    def __init__(self):
        super().__init__('autonomous_navigator')

        # Create publishers and subscribers
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # TF buffer for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation parameters
        self.target_x = 0.0
        self.target_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Navigation state
        self.nav_state = 'STOP'  # STOP, FORWARD, TURN, AVOID
        self.obstacle_threshold = 0.5  # meters

        # Control timers
        self.nav_timer = self.create_timer(0.1, self.navigation_callback)
        self.safety_timer = self.create_timer(0.05, self.safety_check)

        # Set a target (example)
        self.set_target(5.0, 5.0)  # Move to (5m, 5m)

    def set_target(self, x, y):
        """Set navigation target"""
        self.target_x = x
        self.target_y = y
        self.nav_state = 'TURN'  # Need to turn toward target first
        self.get_logger().info(f'Set target to ({x}, {y})')

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Find minimum distance in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 30:len(msg.ranges)//2 + 30]
        self.min_front_dist = min([r for r in front_scan if not math.isnan(r) and r > 0.1])

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        orientation = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        )

    def navigation_callback(self):
        """Main navigation logic"""
        # Calculate angle to target
        target_angle = math.atan2(
            self.target_y - self.current_y,
            self.target_x - self.current_x
        )

        # Calculate angle difference
        angle_diff = target_angle - self.current_yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # State machine for navigation
        if self.nav_state == 'STOP':
            self.stop_robot()

        elif self.nav_state == 'TURN':
            # Turn toward target
            cmd = Twist()

            if abs(angle_diff) > 0.1:  # If not aligned with target
                cmd.angular.z = 0.5 * np.sign(angle_diff)  # Turn toward target
            else:
                # Target aligned, move forward
                self.nav_state = 'FORWARD'

            self.cmd_publisher.publish(cmd)

        elif self.nav_state == 'FORWARD':
            # Move toward target
            cmd = Twist()

            # Check for obstacles
            if self.min_front_dist < self.obstacle_threshold:
                self.nav_state = 'AVOID'
                self.get_logger().info('Obstacle detected, switching to avoidance mode')
            else:
                # Calculate distance to target
                dist_to_target = math.sqrt(
                    (self.target_x - self.current_x)**2 +
                    (self.target_y - self.current_y)**2
                )

                if dist_to_target < 0.3:  # Close enough to target
                    self.nav_state = 'STOP'
                    self.get_logger().info('Reached target location')
                else:
                    cmd.linear.x = 0.3  # Move forward
                    cmd.angular.z = 0.3 * angle_diff  # Correct orientation

            self.cmd_publisher.publish(cmd)

        elif self.nav_state == 'AVOID':
            # Obstacle avoidance behavior
            cmd = Twist()

            # Simple wall-following behavior
            if self.min_front_dist > self.obstacle_threshold:
                self.nav_state = 'FORWARD'
            else:
                # Turn away from obstacle
                cmd.linear.x = 0.1
                cmd.angular.z = 0.3  # Turn right to follow wall

            self.cmd_publisher.publish(cmd)

    def safety_check(self):
        """Safety checks to prevent collisions"""
        if self.min_front_dist < 0.2:  # Emergency stop threshold
            cmd = Twist()
            self.cmd_publisher.publish(cmd)
            self.nav_state = 'STOP'
            self.get_logger().warn('Emergency stop: obstacle too close!')

    def stop_robot(self):
        """Send stop command to robot"""
        cmd = Twist()
        self.cmd_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = AutonomousNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.stop_robot()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Navigation Features

To enhance your autonomous navigation robot:

**SLAM Integration**
- Implement simultaneous localization and mapping
- Use ORB-SLAM, RTAB-MAP, or similar packages
- Build and maintain maps of the environment

**Path Planning**
- Implement A* or D* path planning algorithms
- Consider dynamic obstacles in path planning
- Plan for multi-goal missions

**Machine Learning Enhancement**
- Use reinforcement learning for navigation
- Learn optimal paths through experience
- Adapt behavior based on environment characteristics

![](../../static/img/diagrams/autonomous-navigation-robot.png)

Figure 7.4: Autonomous navigation robot using sensors to build a map and plan paths to navigate to goals.

## Project 5: Smart Home Automation

This project demonstrates how AI and robotics concepts can be applied to automate tasks in a home environment.

### Project Overview

The smart home automation project integrates multiple sensors and actuators to create an intelligent environment that responds to conditions and user preferences. This project teaches systems integration and intelligent decision-making.

### Required Components

- **Central Controller**: Raspberry Pi or similar
- **Sensors**: Temperature, humidity, light, motion, door/window sensors
- **Actuators**: Relays for lights/appliances, servo motors for blinds
- **Communication**: WiFi or Ethernet for device connectivity
- **User Interface**: Mobile app or web dashboard

### Implementation Example

```python
import time
import json
import requests
from datetime import datetime, timedelta
import numpy as np

class SmartHomeController:
    def __init__(self):
        # Initialize sensor readings
        self.sensors = {
            'temperature': 22.0,  # Celsius
            'humidity': 50.0,     # %
            'light_level': 300,   # Lux
            'motion_detected': False,
            'front_door': 'closed',  # 'open' or 'closed'
            'blinds_position': 50    # 0-100%
        }

        # Smart home settings
        self.settings = {
            'target_temp_day': 22.0,
            'target_temp_night': 18.0,
            'light_threshold': 200,
            'security_enabled': True
        }

        # Schedules
        self.schedule = {
            'day_start': 7,   # 7 AM
            'night_start': 22  # 10 PM
        }

        # Learning parameters for adaptive behavior
        self.user_preferences = {
            'preferred_temp': 22.0,
            'preferred_light': 400,
            'typical_wake_time': 7,
            'typical_bed_time': 23
        }

    def read_sensors(self):
        """Simulate reading sensor values"""
        # In real implementation, read from actual sensors
        # This is a simulation with some random variation
        self.sensors['temperature'] += np.random.normal(0, 0.2)
        self.sensors['humidity'] += np.random.normal(0, 0.5)
        self.sensors['light_level'] += np.random.normal(0, 20)

        # Simulate motion detection randomly
        if np.random.random() < 0.1:  # 10% chance
            self.sensors['motion_detected'] = True
        else:
            self.sensors['motion_detected'] = False

    def is_daytime(self):
        """Check if it's currently daytime"""
        current_hour = datetime.now().hour
        return self.schedule['day_start'] <= current_hour < self.schedule['night_start']

    def adjust_climate(self):
        """Adjust heating/cooling based on temperature"""
        is_day = self.is_daytime()
        target_temp = self.settings['target_temp_day'] if is_day else self.settings['target_temp_night']

        if self.sensors['temperature'] < target_temp - 0.5:
            # Need to heat
            print("Heating system activated")
            self.actuate_heater(True)
        elif self.sensors['temperature'] > target_temp + 0.5:
            # Need to cool
            print("Cooling system activated")
            self.actuate_cooler(True)
        else:
            # Temperature is within range
            self.actuate_heater(False)
            self.actuate_cooler(False)

    def adjust_lighting(self):
        """Adjust lighting based on light level and presence"""
        if self.sensors['motion_detected'] and self.sensors['light_level'] < self.settings['light_threshold']:
            # Someone is present and it's dark, turn on lights
            print("Turning on lights")
            self.actuate_lights(True)
        elif not self.sensors['motion_detected']:
            # No one present, turn off unnecessary lights
            print("Turning off lights in unoccupied areas")
            self.actuate_lights(False)

    def manage_blinds(self):
        """Automatically adjust blinds based on light and temperature"""
        if self.is_daytime():
            if self.sensors['light_level'] > 800 and self.sensors['temperature'] > 25.0:
                # Too bright and hot, close blinds
                print("Closing blinds due to high light and temperature")
                self.adjust_blinds(30)  # 30% open
            elif self.sensors['light_level'] < 200:
                # Too dark, open blinds
                print("Opening blinds to let in more light")
                self.adjust_blinds(80)  # 80% open
        else:
            # At night, close blinds for privacy and insulation
            print("Closing blinds for night")
            self.adjust_blinds(10)  # 10% open

    def security_check(self):
        """Monitor security and respond to events"""
        if self.settings['security_enabled']:
            if self.sensors['front_door'] == 'open' during night hours:
                if not self.is_daytime():
                    print("Security alert: Front door opened during night")
                    self.trigger_security_alert()

            if not self.sensors['motion_detected'] but lights are on:
                # Lights on but no motion detected - turn off
                print("No motion detected, turning off unnecessary lights")
                self.actuate_lights(False)

    def learn_user_preferences(self):
        """Learn and adapt to user behavior"""
        current_time = datetime.now()

        # Learn wake time by detecting consistent morning activity
        if 6 <= current_time.hour <= 9 and self.sensors['motion_detected']:
            # User is active in morning, update learned wake time
            self.user_preferences['typical_wake_time'] = current_time.hour

        # Learn bedtime by detecting inactivity in evening
        if 21 <= current_time.hour <= 23 and not self.sensors['motion_detected']:
            # User is inactive in evening, update learned bed time
            self.user_preferences['typical_bed_time'] = current_time.hour

        # Learn temperature preferences based on user adjustments
        # This would be triggered when user manually adjusts temperature

    def actuate_heater(self, on):
        """Control heating system (simulation)"""
        # In real implementation, would send command to actual heater
        pass

    def actuate_cooler(self, on):
        """Control cooling system (simulation)"""
        # In real implementation, would send command to actual cooler
        pass

    def actuate_lights(self, on):
        """Control lighting system (simulation)"""
        # In real implementation, would send command to actual lights
        pass

    def adjust_blinds(self, position):
        """Adjust blind position (0-100%)"""
        # In real implementation, would send command to blind motors
        self.sensors['blinds_position'] = position

    def trigger_security_alert(self):
        """Trigger security alert (simulation)"""
        print("Security system activated - alert sent to user")
        # In real implementation, send notification to user

    def run_cycle(self):
        """Execute one cycle of smart home automation"""
        # Read all sensors
        self.read_sensors()

        # Apply learned intelligence to adjust settings
        self.learn_user_preferences()

        # Execute all control actions
        self.adjust_climate()
        self.adjust_lighting()
        self.manage_blinds()
        self.security_check()

        # Log current state
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}, "
              f"Temp: {self.sensors['temperature']:.1f}C, "
              f"Motion: {self.sensors['motion_detected']}, "
              f"Front door: {self.sensors['front_door']}")

    def run_continuous(self):
        """Run the smart home system continuously"""
        print("Starting Smart Home Automation System")
        try:
            while True:
                self.run_cycle()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("Smart Home Automation System stopped")

# Example usage
# controller = SmartHomeController()
# controller.run_continuous()
```

## Project 6: Edge AI Application

This project demonstrates how to run AI models directly on robotics hardware for real-time decision making.

### Project Overview

Edge AI applications run directly on the robot's hardware without requiring cloud connectivity, enabling real-time processing and reducing latency. This project shows how to deploy AI models for inference on resource-constrained devices.

### Implementation Example

```python
# Edge AI application using TensorFlow Lite
import tensorflow as tf
import numpy as np
import cv2
import time

class EdgeAIProcessor:
    def __init__(self, model_path):
        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape']

        print(f"Model loaded with input shape: {self.input_shape}")

    def preprocess_input(self, raw_input):
        """Preprocess raw input for the model"""
        # Resize if needed
        if self.input_shape[1] != raw_input.shape[0] or self.input_shape[2] != raw_input.shape[1]:
            raw_input = cv2.resize(raw_input, (self.input_shape[1], self.input_shape[2]))

        # Normalize if required (for image models)
        if raw_input.dtype == np.uint8:
            raw_input = raw_input.astype(np.float32) / 255.0

        # Add batch dimension
        processed_input = np.expand_dims(raw_input, axis=0)

        return processed_input

    def run_inference(self, input_data):
        """Run inference on the model"""
        # Preprocess input
        processed_input = self.preprocess_input(input_data)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_input)

        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output, inference_time

    def classify_object(self, image):
        """Classify an object in an image"""
        output, inference_time = self.run_inference(image)

        # Assuming single label classification
        predicted_class = np.argmax(output[0])
        confidence = output[0][predicted_class]

        return predicted_class, confidence, inference_time

class RobotEdgeAIController:
    def __init__(self):
        # Initialize the Edge AI processor
        # In practice, you would load your specific model file
        # For this example, we'll use a placeholder
        try:
            self.ai_processor = EdgeAIProcessor("robot_model.tflite")
            self.ai_enabled = True
            print("Edge AI model loaded successfully")
        except:
            print("Failed to load Edge AI model, running in simulation mode")
            self.ai_enabled = False

    def process_sensor_data(self, sensor_data):
        """Process sensor data using Edge AI"""
        if not self.ai_enabled:
            # Simulation mode
            return self.simulate_ai_processing(sensor_data)

        # Run actual AI processing
        # This could be image processing, sensor fusion, etc.
        if 'camera' in sensor_data:
            image = sensor_data['camera']
            predicted_class, confidence, inference_time = self.ai_processor.classify_object(image)

            print(f"AI classification: Class {predicted_class}, Confidence: {confidence:.2f}, "
                  f"Inference time: {inference_time:.3f}s")

            return {
                'classification': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time,
                'needs_attention': confidence > 0.7  # Object needs attention if confidence > 70%
            }
        else:
            # Process other sensor data
            return self.process_other_sensors(sensor_data)

    def simulate_ai_processing(self, sensor_data):
        """Simulate AI processing when model is not available"""
        # Simulate processing time
        time.sleep(0.05)  # 50ms simulation delay

        # Return simulated results
        return {
            'classification': np.random.randint(0, 10),
            'confidence': np.random.uniform(0.5, 0.9),
            'inference_time': 0.05,
            'needs_attention': np.random.random() > 0.3
        }

    def process_other_sensors(self, sensor_data):
        """Process non-camera sensors"""
        # For other sensors, we might run different AI models
        # or simple rules-based processing
        results = {}

        if 'lidar' in sensor_data:
            # Example: detect obstacles using LIDAR data
            distances = sensor_data['lidar']
            closest_obstacle = min(distances) if distances else float('inf')
            results['closest_obstacle'] = closest_obstacle
            results['obstacle_detected'] = closest_obstacle < 1.0  # 1 meter threshold

        if 'imu' in sensor_data:
            # Example: detect robot orientation
            orientation = sensor_data['imu']
            results['orientation'] = orientation

        return results

    def make_decision(self, ai_results):
        """Make robot decisions based on AI results"""
        if ai_results.get('needs_attention', False):
            # If an object needs attention, move toward it
            print("Object detected, moving closer for inspection")
            return {
                'action': 'move_toward_object',
                'details': f"Object class {ai_results.get('classification', 'unknown')} "
                          f"with confidence {ai_results.get('confidence', 0):.2f}"
            }
        elif ai_results.get('obstacle_detected', False):
            # If obstacle detected, avoid it
            print("Obstacle detected, planning avoidance maneuver")
            return {
                'action': 'avoid_obstacle',
                'details': f"Obstacle at {ai_results.get('closest_obstacle', 'unknown')} meters"
            }
        else:
            # Continue with current behavior
            return {
                'action': 'continue_current_task',
                'details': 'No significant events detected'
            }

# Example usage
def simulate_robot_cycle():
    """Simulate a robot sensing and decision-making cycle"""
    controller = RobotEdgeAIController()

    # Simulate sensor data
    # In a real robot, this would come from actual sensors
    import numpy as np
    dummy_camera = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # 224x224 RGB image
    dummy_lidar = [2.0, 1.5, 3.0, 2.5, 1.8]  # Simulated distances
    dummy_imu = [0.1, 0.2, 0.3]  # Simulated orientation

    sensor_data = {
        'camera': dummy_camera,
        'lidar': dummy_lidar,
        'imu': dummy_imu
    }

    # Process sensor data with AI
    ai_results = controller.process_sensor_data(sensor_data)

    # Make decisions based on AI results
    decision = controller.make_decision(ai_results)

    print(f"Decision: {decision['action']}")
    print(f"Details: {decision['details']}")

    return decision

# Run example
if __name__ == "__main__":
    print("Running Edge AI Robot Controller Example")
    result = simulate_robot_cycle()
```

## Project Planning and Troubleshooting Guides

### General Project Planning

1. **Requirements Definition**
   - Clearly define what your robot should accomplish
   - Identify constraints (budget, size, power, time)
   - Determine success metrics

2. **Component Selection**
   - Choose appropriate hardware for project requirements
   - Consider power consumption, processing requirements, and accuracy needs
   - Verify component compatibility

3. **Development Phases**
   - Start with simple functionality and add complexity
   - Test components individually before integration
   - Implement basic safety mechanisms from the start

4. **Testing Strategy**
   - Plan tests for each component and integration level
   - Consider edge cases and failure scenarios
   - Test in controlled environments before complex scenarios

### Troubleshooting Common Issues

**Sensor Noise and Calibration**
```python
def calibrate_sensor(sensor_data, reference_value):
    """Calibrate sensor readings against known reference"""
    offset = np.mean(sensor_data) - reference_value
    calibrated_data = sensor_data - offset
    return calibrated_data

def filter_sensor_data(raw_data, filter_type='moving_average', window_size=5):
    """Apply filtering to reduce sensor noise"""
    if filter_type == 'moving_average':
        return np.convolve(raw_data, np.ones(window_size)/window_size, mode='same')
    elif filter_type == 'median':
        return scipy.signal.medfilt(raw_data, kernel_size=window_size)
    else:
        return raw_data
```

**Motor Control Issues**
```python
def safe_motor_control(target_speed, current_speed, max_change_rate=0.1):
    """Limit acceleration/deceleration to protect motors"""
    speed_diff = target_speed - current_speed
    if abs(speed_diff) > max_change_rate:
        # Limit the rate of change
        target_speed = current_speed + max_change_rate * np.sign(speed_diff)
    return target_speed
```

**System Integration Problems**
- Use modular design to isolate issues
- Implement logging to track system state
- Test communication between modules separately

## Assessment Methodologies

To evaluate the successful completion of your projects, consider these assessment approaches:

1. **Functional Testing**
   - Test each feature individually
   - Verify performance meets requirements
   - Document any limitations or issues

2. **Robustness Testing**
   - Test with various input conditions
   - Verify behavior under stress conditions
   - Check for safe failure modes

3. **Performance Metrics**
   - Measure execution time
   - Check power consumption
   - Evaluate accuracy of AI components

4. **Documentation Review**
   - Verify code is well-commented
   - Check that the build process is documented
   - Ensure safety considerations are addressed

Each project should include a project report that outlines:
- Design decisions and rationale
- Challenges encountered and solutions
- Performance evaluation
- Future improvement opportunities

---

## Chapter Summary

In this chapter, we explored several comprehensive projects that integrate the concepts from previous chapters into practical applications. We covered line-following robots, object-detection systems, voice-controlled robots, autonomous navigation, smart home automation, and edge AI applications. Each project builds complexity and demonstrates how to apply AI and robotics concepts in real-world implementations.

## Quiz: Chapter 7

1. In the PID controller for the line-following robot, what does the 'P' term represent?

   a) Power consumption

   b) Proportional response to current error

   c) Processing time

   d) Position measurement

2. What is the main purpose of using TensorFlow Lite in robotics projects?

   a) Training neural networks

   b) Running inference on edge devices with limited resources

   c) Storing sensor data

   d) Communicating with other robots

3. Which sensor would be most appropriate for a reliable line-following robot?

   a) Temperature sensor

   b) Gyroscope

   c) Optical/IR sensors

   d) Accelerometer

4. What does SLAM stand for in robotics?

   a) Systematic Localization and Mapping

   b) Simultaneous Localization and Mapping

   c) Sensor Learning and Mapping

   d) Stereo Localization and Mapping

5. True or False: In the voice-controlled robot example, the speech recognition happens offline without internet connectivity.

   a) True

   b) False

6. For the autonomous navigation robot, what is the purpose of the safety timer?

   a) To calibrate sensors

   b) To prevent collisions by monitoring obstacle distances

   c) To update the map

   d) To control movement speed

7. Which of the following is NOT typically part of a smart home automation system?

   a) Temperature sensors

   b) Motion detectors

   c) Robot actuators

   d) Web dashboard

8. What is the main advantage of edge AI over cloud AI in robotics applications?

   a) More processing power

   b) Lower latency and improved reliability

   c) Larger model sizes

   d) Easier updates

9. In the object-detection robot project, what does the system do when it identifies an object of interest?

   a) Immediately picks up the object

   b) Approaches the object while keeping it centered in the camera

   c) Takes a photo and stops

   d) Reports the object to a cloud service

10. Which of the following is an important consideration when planning a robotics project?

    a) Budget and component compatibility

    b) Development phases and testing strategy

    c) Safety mechanisms

    d) All of the above