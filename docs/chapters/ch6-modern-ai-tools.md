---
sidebar_label: 'Chapter 6: Modern AI Development Tools for Robotics'
sidebar_position: 7
---

# Chapter 6: Modern AI Development Tools for Robotics

## Python for Robotics

Python has become the de facto standard for robotics and AI development due to its simplicity, extensive libraries, and strong community support. Its readability makes it ideal for prototyping and collaboration in robotics projects.

### Why Python for Robotics?

**Rich Ecosystem**: Python has libraries for every aspect of robotics:
- NumPy and Pandas for numerical computing and data manipulation
- SciPy for scientific computing
- OpenCV for computer vision
- Matplotlib for visualization
- ROS bridge libraries for Robot Operating System

**Prototyping Speed**: Python's concise syntax allows rapid development and testing of robotic algorithms.

**Cross-Platform Compatibility**: Python runs on various platforms from Raspberry Pi to powerful AI workstations.

**Community Support**: Large communities developing and sharing robotics-specific packages.

### Essential Python Libraries for Robotics

**NumPy**: Fundamental package for numerical computing
```python
import numpy as np

# Working with robot position data (x, y, z coordinates)
position = np.array([1.0, 2.0, 0.5])
velocity = np.array([0.1, 0.0, 0.0])

# Calculate new position after time step
dt = 0.1
new_position = position + velocity * dt
print(f"New position: {new_position}")
```

**SciPy**: For scientific computing and optimization
```python
from scipy.spatial.transform import Rotation as R
import numpy as np

# Robot orientation using quaternions
quat = np.array([0.707, 0, 0, 0.707])  # 90-degree rotation about Z-axis
r = R.from_quat(quat)
rotation_matrix = r.as_matrix()
print("Rotation matrix:")
print(rotation_matrix)
```

**OpenCV**: For computer vision tasks in robotics
```python
import cv2
import numpy as np

# Reading and processing robot camera feed
def detect_obstacles(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours (potential obstacles)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original frame
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result, len(contours)  # Return image with contours and count

# Use in robot control loop
# camera_feed = get_camera_image()
# processed_image, obstacle_count = detect_obstacles(camera_feed)
```

### Python Best Practices for Robotics

**Error Handling**: Robotics applications must handle unexpected situations gracefully:
```python
import time
import logging

def safe_motor_command(motor_id, speed):
    try:
        # Validate input
        if not -100 <= speed <= 100:
            raise ValueError(f"Speed {speed} out of range [-100, 100]")

        # Send command to motor
        send_command_to_motor(motor_id, speed)

    except ValueError as e:
        logging.error(f"Invalid command for motor {motor_id}: {e}")
    except ConnectionError:
        logging.error(f"Cannot connect to motor {motor_id}")
    except Exception as e:
        logging.error(f"Unexpected error with motor {motor_id}: {e}")
```

**Threading and Asynchronous Programming**:
```python
import threading
import queue
import time

# Handling multiple sensors simultaneously
class SensorManager:
    def __init__(self):
        self.sensor_data = {}
        self.data_queue = queue.Queue()
        self.running = True

    def read_sensors(self):
        while self.running:
            # Simulate reading different sensors
            sensor_data = {
                'camera': get_camera_data(),
                'lidar': get_lidar_data(),
                'imu': get_imu_data()
            }

            self.data_queue.put(sensor_data)
            time.sleep(0.1)  # 10Hz

    def start(self):
        sensor_thread = threading.Thread(target=self.read_sensors)
        sensor_thread.start()
        return sensor_thread

# Usage
sensor_manager = SensorManager()
sensor_thread = sensor_manager.start()

while True:
    try:
        data = sensor_manager.data_queue.get(timeout=1.0)
        # Process sensor data
        process_robot_behavior(data)
    except queue.Empty:
        print("Sensor timeout - robot may need to stop")
        stop_robot_safely()
```

## PyTorch vs TensorFlow for Robotics

Both PyTorch and TensorFlow are leading deep learning frameworks with distinct advantages for robotics applications. Understanding their differences helps you choose the right tool for your specific needs.

### PyTorch: Pythonic Deep Learning

**Eager Execution**: PyTorch executes operations immediately, making it intuitive for debugging and experimentation.
```python
import torch
import torch.nn as nn

# Define a simple neural network for robot control
class RobotController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RobotController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create and test the model
model = RobotController(input_size=4, hidden_size=64, output_size=2)  # 4 inputs (sensor readings), 2 outputs (motor commands)

# Example sensor inputs
sensor_inputs = torch.tensor([[1.0, 0.5, 2.0, -0.5]], dtype=torch.float32)

# Get motor commands
motor_outputs = model(sensor_inputs)
print(f"Motor commands: {motor_outputs}")
```

**Dynamic Computation Graphs**: PyTorch builds computation graphs dynamically, making it suitable for research and complex models with variable structures.

**Python Integration**: PyTorch feels natural to Python developers with its object-oriented design.

### TensorFlow: Production-Ready Framework

**Graph-Based Execution**: TensorFlow constructs a computational graph before execution, enabling optimization and deployment.
```python
import tensorflow as tf

# Define the same robot controller in TensorFlow
class RobotController(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(RobotController, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

# Create and test the model
model = RobotController(input_size=4, hidden_size=64, output_size=2)

# Example sensor inputs
sensor_inputs = tf.constant([[1.0, 0.5, 2.0, -0.5]], dtype=tf.float32)

# Get motor commands
motor_outputs = model(sensor_inputs)
print(f"Motor commands: {motor_outputs}")
```

**TensorFlow Lite**: Specifically designed for mobile and embedded deployment, crucial for robotics.
```python
# Convert model for edge deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('robot_controller.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Comparing PyTorch and TensorFlow for Robotics

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| Learning Curve | Gentler, more Pythonic | Steeper initially |
| Production Deployment | Improving with TorchScript | Excellent with TensorFlow Lite/Serving |
| Mobile/Edge Deployment | TorchScript, LibTorch | TensorFlow Lite |
| Visualization | Limited native tools | TensorBoard |
| Community | Strong in research | Strong in production |
| Performance | Good | Excellent with optimization |

### Choosing Between PyTorch and TensorFlow

**Choose PyTorch when:**
- Prototyping and research
- Need fast iteration and debugging
- Working with complex, variable architectures
- Team is more familiar with Python's ecosystem

**Choose TensorFlow when:**
- Production deployment is the primary goal
- Target edge devices (TF Lite)
- Need extensive model optimization
- Working in enterprise environments

### Practical Example: Training a Robot Controller

Here's how you might train a neural network controller in both frameworks:

**PyTorch Training Example:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Robot simulation environment (simplified)
def robot_simulation(sensor_inputs, action):
    # Simplified robot physics
    # In reality, this might connect to a simulator or real robot
    next_state = sensor_inputs + action * 0.1  # Simplified physics
    reward = -torch.sum(torch.abs(next_state - torch.tensor([0.0, 0.0, 0.0, 0.0])))  # Negative distance from target
    return next_state, reward

# Define model, loss, and optimizer
model = RobotController(input_size=4, hidden_size=64, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(1000):
    # Sample random initial state
    state = torch.randn(1, 4)

    # Forward pass
    action = model(state)

    # Simulate robot response
    next_state, reward = robot_simulation(state, action)

    # Create target action (would come from expert demonstrations in real scenario)
    target_action = torch.tensor([[0.0, 0.0]])  # Simplified target

    # Calculate loss (in real scenario, this would be more complex)
    loss = criterion(action, target_action)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

**TensorFlow Training Example:**
```python
import tensorflow as tf
import numpy as np

# Robot simulation environment
def robot_simulation(state, action):
    # Simplified physics model
    next_state = state + action * 0.1
    reward = -tf.reduce_sum(tf.square(next_state))  # Negative distance from target
    return next_state, reward

# Define model
model = RobotController(input_size=4, hidden_size=64, output_size=2)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
for epoch in range(1000):
    # Sample random initial state
    state = tf.random.normal((1, 4))

    with tf.GradientTape() as tape:
        # Forward pass
        action = model(state)

        # Simulate robot response
        next_state, reward = robot_simulation(state, action)

        # Calculate loss (simplified)
        target_action = tf.zeros((1, 2))  # Simplified target
        loss = tf.keras.losses.mse(action, target_action)

    # Calculate gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy():.4f}')
```

![](../../static/img/diagrams/pytorch-tensorflow-comparison.png)

Figure 6.1: Comparison of PyTorch and TensorFlow for robotics applications highlighting their different strengths and use cases.

## Robot Operating System (ROS 2) Basics

ROS (Robot Operating System) is not an actual operating system but a middleware framework that provides services for robot software development. ROS 2 is the newer, improved version designed to address limitations in the original ROS.

### Key Concepts in ROS 2

**Nodes**: Individual processes that perform specific functions. In a robot system, you might have nodes for:
- Sensor drivers
- Control algorithms
- Perception systems
- Planning modules
- User interfaces

**Topics**: Communication channels for passing messages between nodes. Topics use a publish-subscribe pattern:
- Publisher nodes send messages to topics
- Subscriber nodes receive messages from topics
- Multiple publishers and subscribers can use the same topic

**Services**: Communication pattern for request-response interactions:
- Service clients send requests
- Service servers provide responses
- Useful for operations that require acknowledgment

### Setting up ROS 2

ROS 2 requires a Linux-based system (though Windows and macOS support exists). The most common installation is on Ubuntu with one of the supported distributions.

### Creating a Simple ROS 2 Package

```bash
# Create a new package
ros2 pkg create --build-type ament_python my_robot_controller

cd my_robot_controller
```

### ROS 2 Python Example

A simple publisher node for sensor data:

```python
# my_robot_controller/my_robot_controller/sensor_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'sensor_data', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Simulate sensor readings
        msg = Float64MultiArray()
        msg.data = [
            random.uniform(0, 10),   # Distance reading 1
            random.uniform(0, 10),   # Distance reading 2
            random.uniform(-1, 1),   # Gyro reading
            random.uniform(-1, 1)    # Accelerometer reading
        ]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

A corresponding subscriber node for processing sensor data:

```python
# my_robot_controller/my_robot_controller/obstacle_detector.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'sensor_data',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Simple obstacle detection: if any distance sensor < 1.0, turn
        sensor_data = msg.data
        obstacle_detected = any(dist < 1.0 for dist in sensor_data[:2])

        cmd_msg = Twist()
        if obstacle_detected:
            # Turn in place
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5
        else:
            # Move forward
            cmd_msg.linear.x = 0.5
            cmd_msg.angular.z = 0.0

        self.publisher_.publish(cmd_msg)
        self.get_logger().info(f'Published cmd_vel: linear={cmd_msg.linear.x}, angular={cmd_msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = ObstacleDetector()

    try:
        rclpy.spin(obstacle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### ROS 2 for Physical AI Integration

ROS 2 excels at integrating different AI components in robotic systems:

1. **Modularity**: Each AI component can be its own node
2. **Reusability**: Standardized interfaces allow component reuse
3. **Simulation**: Easy integration with simulation environments like Gazebo
4. **Tooling**: Extensive debugging and visualization tools
5. **Distributed Computing**: Nodes can run on different machines

### ROS 2 vs Other Frameworks

| Aspect | ROS 2 | Alternative |
|--------|-------|-------------|
| Modularity | Excellent | Custom frameworks require building from scratch |
| Community | Very large | Proprietary solutions are limited |
| Simulation | Gazebo integration | Other simulators require custom integration |
| Hardware Support | Extensive packages | Limited support without custom development |
| Learning Curve | Moderate to steep | Varies by approach |

![](../../static/img/diagrams/ros2-concept.png)

Figure 6.2: Key concepts in ROS 2 including nodes, topics, and services for robot software architecture.

## ONNX and Model Deployment

ONNX (Open Neural Network Exchange) is a standard format for representing machine learning models that enables models to be transferred between different frameworks. This is crucial for robotics where you might train in one framework and deploy on another.

### Why ONNX Matters for Robotics

**Framework Interoperability**: Train in PyTorch, TensorFlow, or other frameworks, then deploy using ONNX-compatible runtimes.

**Optimization**: ONNX provides tools for optimizing models for deployment.

**Edge Deployment**: Many embedded and edge platforms support ONNX for AI inference.

**Hardware Acceleration**: ONNX models can leverage specialized hardware more easily.

### Converting Models to ONNX

**Converting PyTorch to ONNX:**
```python
import torch
import torch.onnx

# Define and create your model
class RobotController(nn.Module):
    def __init__(self):
        super(RobotController, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = RobotController()
model.eval()  # Set to evaluation mode

# Sample input for tracing
sample_input = torch.randn(1, 4)

# Export to ONNX
torch.onnx.export(
    model,                    # Model to export
    sample_input,             # Model input (or a tuple of multiple inputs)
    "robot_controller.onnx",  # Output file name
    export_params=True,       # Store trained parameter weights
    opset_version=11,         # ONNX version to export to
    do_constant_folding=True, # Execute constant folding for optimization
    input_names=['sensor_input'],   # Model's input names
    output_names=['motor_output'],  # Model's output names
    dynamic_axes={
        'sensor_input': {0: 'batch_size'},    # Variable length axes
        'motor_output': {0: 'batch_size'}
    }
)

print("Model exported to robot_controller.onnx")
```

**Converting TensorFlow to ONNX:**
```python
import tf2onnx
import tensorflow as tf

# Convert using tf2onnx converter
spec = (tf.TensorSpec((None, 4), tf.float32, name="sensor_input"),)
output_path = "robot_controller.onnx"

# Convert the model
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)

# Save the model
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model converted and saved to {output_path}")
```

### Using ONNX Models in Robotics

**ONNX Runtime for Inference:**
```python
import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("robot_controller.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
sensor_input = np.array([[1.0, 0.5, 2.0, -0.5]], dtype=np.float32)
motor_output = session.run([output_name], {input_name: sensor_input})

print(f"Motor command: {motor_output[0][0]}")
```

### Optimizing ONNX Models

ONNX provides tools for model optimization:

```python
import onnx
from onnx import optimizer

# Load model
model = onnx.load("robot_controller.onnx")

# List of optimization passes
passes = [
    'eliminate_deadend',      # Remove unreachable parts
    'eliminate_identity',     # Remove identity operations
    'eliminate_nop_dropout',  # Remove no-op dropout nodes
    'eliminate_nop_transpose', # Remove no-op transpose nodes
    'fuse_add_bias_into_conv', # Fuse add bias into convolutions
    'fuse_bn_into_conv'       # Fuse batch norm into convolutions
]

# Apply optimizations
optimized_model = optimizer.optimize(model, passes)

# Save optimized model
onnx.save(optimized_model, "robot_controller_optimized.onnx")
```

### ONNX in Edge Robotics

ONNX is particularly valuable for edge robotics because many hardware platforms support ONNX runtimes:

- **NVIDIA Jetson**: TensorRT for optimized ONNX inference
- **Raspberry Pi**: ONNX Runtime for ARM
- **Mobile**: ONNX Runtime Mobile
- **Intel**: OpenVINO supports ONNX models

![](../../static/img/diagrams/onnx-flow.png)

Figure 6.3: ONNX enables model transfer between different AI frameworks and deployment platforms.

## OpenAI Agents and Robotics Pipelines

OpenAI's tools, particularly their API and agents, are increasingly being integrated into robotics pipelines for higher-level decision making and control.

### OpenAI API for Robotics

The OpenAI API can be used in robotics applications for tasks requiring natural language processing, high-level planning, or complex reasoning:

```python
import openai
import json

# Example: Natural language command interpretation
def interpret_command(natural_language_command):
    prompt = f"""
    You are a robot command interpreter. Convert the following natural language command into a structured format that a robot can execute.

    Command: "{natural_language_command}"

    Output format as JSON:
    {{
        "action": "...",
        "parameters": {{
            "target_location": "...",
            "object": "...",
            "gripper_position": "..."
        }},
        "priority": "..."
    }}

    Examples:
    Input: "Go to the kitchen and get the red apple"
    Output: {{
        "action": "navigate_and_pick",
        "parameters": {{
            "target_location": "kitchen",
            "object": "red apple"
        }},
        "priority": "high"
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" for better performance
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    try:
        result = json.loads(response.choices[0].message['content'])
        return result
    except json.JSONDecodeError:
        # Handle case where response isn't valid JSON
        print("Failed to parse response as JSON")
        return None

# Example usage
command = "Move to the living room and bring me the book from the table"
structured_command = interpret_command(command)
print(f"Interpreted command: {structured_command}")
```

### Robotics Pipeline with OpenAI Integration

A complete robotics pipeline might integrate OpenAI tools at different levels:

```python
import openai
import json
from typing import Dict, List, Any

class RobotAIAgent:
    def __init__(self):
        self.location_map = {
            "kitchen": [1.0, 2.0, 0.0],
            "living room": [5.0, 0.0, 0.0],
            "bedroom": [8.0, 3.0, 0.0],
            "office": [0.0, 5.0, 0.0]
        }

    def interpret_task(self, user_request: str) -> Dict[str, Any]:
        """Use OpenAI to interpret high-level tasks into structured commands"""
        prompt = f"""
        You are a household robot task planner. Interpret the user's request into a sequence of structured commands.

        User request: "{user_request}"

        Output a list of structured commands in JSON format:
        [
            {{
                "command": "navigate" | "detect" | "pick" | "place" | "speak",
                "parameters": {{
                    "location": "...",
                    "object": "...",
                    "message": "..."
                }}
            }}
        ]

        Example:
        User: "Please bring me a drink from the kitchen"
        Output: [
            {{
                "command": "navigate",
                "parameters": {{
                    "location": "kitchen"
                }}
            }},
            {{
                "command": "detect",
                "parameters": {{
                    "object": "drink"
                }}
            }},
            {{
                "command": "pick",
                "parameters": {{
                    "object": "drink"
                }}
            }},
            {{
                "command": "navigate",
                "parameters": {{
                    "location": "user"
                }}
            }},
            {{
                "command": "place",
                "parameters": {{
                    "location": "user"
                }}
            }}
        ]
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        try:
            commands = json.loads(response.choices[0].message['content'])
            return commands
        except json.JSONDecodeError:
            return [{"command": "speak", "parameters": {"message": "Sorry, I couldn't understand your request."}}]

    def execute_command(self, command: Dict[str, Any]) -> bool:
        """Execute a single command (simulation)"""
        cmd_type = command["command"]
        params = command.get("parameters", {})

        print(f"Executing: {cmd_type} with params {params}")

        # In a real robot, this would interface with the robot's control systems
        # For simulation, we just return success
        if cmd_type == "navigate":
            if params.get("location") in self.location_map:
                print(f"Navigating to {params['location']}")
                return True
            else:
                print(f"Unknown location: {params['location']}")
                return False
        elif cmd_type == "detect":
            print(f"Detecting {params.get('object', 'objects')}")
            # In real implementation, this would use computer vision
            return True
        elif cmd_type == "pick":
            print(f"Picking {params.get('object', 'object')}")
            return True
        elif cmd_type == "place":
            print(f"Placing object at {params.get('location', 'current location')}")
            return True
        elif cmd_type == "speak":
            print(f"Speaking: {params.get('message', '')}")
            return True
        else:
            print(f"Unknown command: {cmd_type}")
            return False

    def execute_task(self, user_request: str) -> bool:
        """Execute a complete task from user request"""
        print(f"Processing user request: {user_request}")

        # Interpret the task using OpenAI
        commands = self.interpret_task(user_request)
        print(f"Generated commands: {commands}")

        # Execute each command
        for command in commands:
            success = self.execute_command(command)
            if not success:
                print(f"Command failed: {command}")
                return False

        print("Task completed successfully")
        return True

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    # openai.api_key = "your-api-key-here"

    robot = RobotAIAgent()

    # Example task
    user_request = "Please go to the kitchen and bring me a cold drink"
    success = robot.execute_task(user_request)

    if success:
        print("Robot successfully completed the task!")
    else:
        print("Robot failed to complete the task.")
```

### Integration with ROS 2

OpenAI tools can be integrated with ROS 2 systems to create high-level autonomous capabilities:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import openai
import json

class OpenAIRobotController(Node):
    def __init__(self):
        super().__init__('openai_robot_controller')

        # Subscriber for high-level commands
        self.command_subscriber = self.create_subscription(
            String,
            'high_level_command',
            self.command_callback,
            10
        )

        # Publisher for low-level robot commands
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publisher for robot responses
        self.response_publisher = self.create_publisher(String, 'robot_response', 10)

        # Set your OpenAI API key
        # openai.api_key = 'your-api-key-here'

    def command_callback(self, msg):
        """Handle high-level command from user"""
        user_command = msg.data
        self.get_logger().info(f'Received command: {user_command}')

        # Use OpenAI to interpret and plan the command
        try:
            plan = self.generate_plan(user_command)
            self.execute_plan(plan)
        except Exception as e:
            self.get_logger().error(f'Error executing command: {e}')
            error_msg = String()
            error_msg.data = f"Sorry, I couldn't execute the command: {e}"
            self.response_publisher.publish(error_msg)

    def generate_plan(self, user_command):
        """Use OpenAI to generate a plan from natural language"""
        prompt = f"""
        You are a navigation planner for a mobile robot. Convert the user command into a sequence of low-level navigation commands.

        User command: "{user_command}"

        Output your response as a JSON list of navigation commands with:
        {{
            "type": "move_forward" | "turn_left" | "turn_right" | "stop",
            "duration": seconds (float),
            "velocity": m/s (float)
        }}

        Example:
        User: "Go forward 2 meters"
        Output: [{{"type": "move_forward", "duration": 4.0, "velocity": 0.5}}]
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        plan = json.loads(response.choices[0].message['content'])
        return plan

    def execute_plan(self, plan):
        """Execute the planned navigation commands"""
        for step in plan:
            cmd_msg = Twist()

            if step['type'] == 'move_forward':
                cmd_msg.linear.x = step['velocity']
            elif step['type'] == 'turn_left':
                cmd_msg.angular.z = 0.5  # radians per second
            elif step['type'] == 'turn_right':
                cmd_msg.angular.z = -0.5  # radians per second
            elif step['type'] == 'stop':
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0

            # Execute command for specified duration
            duration = step['duration']
            start_time = self.get_clock().now()

            while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
                self.cmd_publisher.publish(cmd_msg)
                rclpy.spin_once(self, timeout_sec=0.1)

            # Stop robot after command
            stop_msg = Twist()
            self.cmd_publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = OpenAIRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

![](../../static/img/diagrams/openai-robotics-pipeline.png)

Figure 6.4: Integration of OpenAI tools in robotics pipeline for high-level task interpretation and planning.

---

## Hands-On Lab: Creating a Simple Robot AI Controller

### Objective
To implement a simple AI controller using Python and simulate its integration with a robot.

### Prerequisites
- Python installed on your system
- Required packages: `pip install numpy matplotlib torch`

### Lab Instructions

1. **Implement the Robot Controller Class**:
   ```python
   import numpy as np
   import torch
   import torch.nn as nn
   import torch.optim as optim
   import matplotlib.pyplot as plt
   from matplotlib.animation import FuncAnimation
   import time

   class RobotController(nn.Module):
       def __init__(self, input_size=4, hidden_size=64, output_size=2):
           super(RobotController, self).__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(hidden_size, output_size)

       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x
   ```

2. **Create a Simple Robot Environment**:
   ```python
   class SimpleRobotEnvironment:
       def __init__(self):
           # Robot starts at origin facing right
           self.position = np.array([0.0, 0.0])
           self.orientation = 0.0  # Angle in radians
           self.velocity = 0.0
           self.angular_velocity = 0.0

           # Target location
           self.target = np.array([5.0, 3.0])

           # Obstacles
           self.obstacles = [
               {'pos': np.array([2.0, 2.0]), 'radius': 0.5},
               {'pos': np.array([3.0, 1.0]), 'radius': 0.5}
           ]

       def get_sensor_data(self):
           """Simulate sensor readings:
           - Distance to target (x, y)
           - Distance to nearest obstacle (x, y)
           - Current orientation
           - Current velocity
           """
           # Distance to target
           dist_to_target = self.target - self.position

           # Find nearest obstacle
           min_dist = float('inf')
           nearest_obstacle_dir = np.array([0.0, 0.0])

           for obs in self.obstacles:
               dist_vec = obs['pos'] - self.position
               dist = np.linalg.norm(dist_vec)

               if dist < min_dist and dist > 0.1:  # Avoid very close obstacles (likely self-detection)
                   min_dist = dist
                   nearest_obstacle_dir = dist_vec / dist  # Normalize direction

           # If no obstacle is detected, set direction to zero
           if min_dist == float('inf'):
               nearest_obstacle_dir = np.array([0.0, 0.0])

           # Return sensor data: [target_dir_x, target_dir_y, obstacle_dir_x, obstacle_dir_y]
           return np.concatenate([dist_to_target, nearest_obstacle_dir[:2]]).astype(np.float32)

       def apply_action(self, action):
           """Apply motor commands to move the robot"""
           linear_vel, angular_vel = action

           # Apply bounds to velocities
           linear_vel = np.clip(linear_vel, -0.5, 0.5)
           angular_vel = np.clip(angular_vel, -0.5, 0.5)

           # Update robot state
           dt = 0.1  # Time step
           self.orientation += angular_vel * dt
           self.position[0] += linear_vel * np.cos(self.orientation) * dt
           self.position[1] += linear_vel * np.sin(self.orientation) * dt

           # Check for collisions with obstacles
           for obs in self.obstacles:
               dist = np.linalg.norm(self.position - obs['pos'])
               if dist < obs['radius'] + 0.2:  # Robot radius of 0.2
                   print("Collision detected!")
                   # Move robot back to previous position
                   self.orientation -= angular_vel * dt
                   self.position[0] -= linear_vel * np.cos(self.orientation) * dt
                   self.position[1] -= linear_vel * np.sin(self.orientation) * dt
                   return -10  # High negative reward for collision

           # Calculate reward (negative distance to target, with bonus for getting close)
           dist_to_target = np.linalg.norm(self.position - self.target)
           reward = -dist_to_target

           # Bonus for reaching target
           if dist_to_target < 0.3:
               reward += 100
               print("Target reached!")

           return reward
   ```

3. **Implement the Training Loop**:
   ```python
   def train_robot_controller():
       env = SimpleRobotEnvironment()
       model = RobotController()
       optimizer = optim.Adam(model.parameters(), lr=0.001)
       criterion = nn.MSELoss()

       # Training parameters
       num_episodes = 500
       max_steps = 100
       total_rewards = []

       for episode in range(num_episodes):
           # Reset environment
           env = SimpleRobotEnvironment()  # Reset position

           total_reward = 0

           for step in range(max_steps):
               # Get sensor data
               sensor_data = env.get_sensor_data()

               # Convert to tensor and get action
               state_tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
               action_tensor = model(state_tensor)
               action = action_tensor.detach().numpy()[0]

               # Apply action to environment
               reward = env.apply_action(action)
               total_reward += reward

               # Check if target reached
               if np.linalg.norm(env.position - env.target) < 0.3:
                   break

               # Early termination if robot gets stuck
               if step > 50 and total_reward < -20:
                   break

           total_rewards.append(total_reward)

           if episode % 50 == 0:
               print(f"Episode {episode}, Average Reward: {np.mean(total_rewards[-50:]) if len(total_rewards) >= 50 else np.mean(total_rewards):.2f}")

       return model, total_rewards
   ```

4. **Run the Training and Visualization**:
   ```python
   def visualize_robot_behavior(model):
       env = SimpleRobotEnvironment()
       positions = [env.position.copy()]

       # Run simulation for visualization
       for step in range(100):
           sensor_data = env.get_sensor_data()
           state_tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
           action_tensor = model(state_tensor)
           action = action_tensor.detach().numpy()[0]

           env.apply_action(action)
           positions.append(env.position.copy())

           if np.linalg.norm(env.position - env.target) < 0.3:
               print(f"Target reached in {step+1} steps!")
               break

       # Plot the robot's path
       positions = np.array(positions)
       obstacles = env.obstacles

       plt.figure(figsize=(10, 8))

       # Plot robot path
       plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Robot Path', linewidth=2)
       plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
       plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
       plt.plot(env.target[0], env.target[1], 'k*', markersize=15, label='Target')

       # Plot obstacles
       for obs in obstacles:
           circle = plt.Circle((obs['pos'][0], obs['pos'][1]), obs['radius'], color='red', alpha=0.5)
           plt.gca().add_patch(circle)

       plt.xlim(-1, 6)
       plt.ylim(-1, 5)
       plt.grid(True, alpha=0.3)
       plt.legend()
       plt.title('Robot Navigation with AI Controller')
       plt.xlabel('X Position')
       plt.ylabel('Y Position')
       plt.show()

   # Train the model
   print("Training robot controller...")
   trained_model, rewards = train_robot_controller()

   # Visualize the trained controller
   print("\nVisualizing trained controller...")
   visualize_robot_behavior(trained_model)

   # Plot training progress
   plt.figure(figsize=(10, 5))
   plt.plot(rewards)
   plt.title('Training Progress')
   plt.xlabel('Episode')
   plt.ylabel('Total Reward')
   plt.grid(True, alpha=0.3)
   plt.show()
   ```

5. **Test the Model**:
   - Run the complete script to train the robot controller
   - Observe how the robot learns to navigate toward the target while avoiding obstacles
   - Experiment with adjusting the neural network architecture or training parameters

### Questions for Reflection
1. How did the robot controller learn to navigate toward the target?
2. What would happen if you added more obstacles or changed their positions?
3. How might you modify the reward function to prioritize obstacle avoidance over reaching the target quickly?
4. What are the limitations of this simple approach, and how might you address them in a real robot?

---

## Chapter Summary

In this chapter, we explored modern AI development tools essential for robotics applications. We covered Python's role in robotics development, compared PyTorch and TensorFlow for different robotics use cases, introduced ROS 2 concepts, discussed ONNX for model deployment, and explored how OpenAI tools can be integrated into robotics pipelines.

## Quiz: Chapter 6

1. Which Python library is most commonly used for computer vision tasks in robotics?

   a) NumPy

   b) Pandas

   c) OpenCV

   d) Matplotlib

2. What is the main difference between PyTorch and TensorFlow execution models?

   a) PyTorch uses dynamic graphs, TensorFlow uses static graphs

   b) PyTorch is faster than TensorFlow

   c) PyTorch has more pre-trained models

   d) There is no significant difference

3. In ROS 2, what is the communication pattern used for real-time sensor data?

   a) Services (request-response)

   b) Actions (goal-result-feedback)

   c) Topics (publish-subscribe)

   d) Parameters (configuration)

4. What does ONNX stand for?

   a) Open Neural Network Exchange

   b) Optimized Neural Network Execution

   c) Open Neural Network Extension

   d) Operational Neural Network Expansion

5. True or False: ONNX enables model interoperability between different deep learning frameworks.

   a) True

   b) False

6. Which ROS 2 communication pattern would be most appropriate for requesting the robot to move to a specific location?

   a) Topics (publish-subscribe)

   b) Services (request-response)

   c) Parameters (configuration)

   d) Actions (goal-result-feedback)

7. What is a key advantage of using TensorFlow Lite for robotics applications?

   a) Better training performance

   b) Optimized for mobile and embedded deployment

   c) More pre-trained models available

   d) Easier to learn than regular TensorFlow

8. In the context of robotics, what does "eager execution" mean in PyTorch?

   a) The model trains faster

   b) Operations are executed immediately when called

   c) The model runs on the CPU by default

   d) The model uses less memory

9. Which OpenAI model would likely be most appropriate for interpreting natural language robot commands?

   a) DALL-E

   b) GPT-3.5 or GPT-4

   c) CLIP

   d) Whisper

10. Which of the following is NOT a benefit of using ONNX in robotics?

    a) Framework interoperability

    b) Model optimization tools

    c) Hardware acceleration support

    d) Elimination of all model accuracy issues