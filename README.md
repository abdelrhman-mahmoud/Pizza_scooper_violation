# Pizza Store Scooper Violation Detection System

This project implements a real-time violation detection system for a pizza store, focusing on identifying instances where hands come into contact with pizza without the use of a scooper. The system leverages computer vision (YOLO for object detection and tracking) and a microservices architecture to process video streams, detect objects, identify violations, and provide a live dashboard.

## Features

*   **Real-time Object Detection:** Utilizes YOLO to detect hands, scoopers, and pizza in video frames.
*   **Hand Tracking:** Tracks individual hands to monitor their movement and interaction with designated areas.
*   **Region of Interest (ROI) Monitoring:** Defines specific areas (e.g., near the pizza) to trigger violation checks.
*   **Scooper Usage Verification:** Checks if a scooper is present when a hand enters the pizza area.
*   **Violation Logging:** Records detected violations, including the frame and bounding box of the violating hand.
*   **Live Streaming Dashboard:** Provides a web interface to view the annotated video stream and real-time violation count.
*   **Microservices Architecture:** Decoupled services for detection, violation logic, and streaming, enabling scalability and maintainability.
*   **RabbitMQ for Inter-service Communication:** Efficient and reliable message queuing for data exchange between services.

## Architecture

The system is composed of three main microservices, orchestrated using Docker Compose:

1.  **`detection-service`**:
    *   **Purpose:** Responsible for processing video input, performing object detection (hands, scoopers, pizza) using a YOLO model, and tracking detected objects.
    *   **Input:** Video stream (currently from a hardcoded video file).
    *   **Output:** Publishes detected objects (including their IDs, labels, and bounding boxes) and the raw video frame to RabbitMQ.

2.  **`violation-service`**:
    *   **Purpose:** Consumes data from the `detection-service` and applies the core violation logic. It determines if a hand has touched the pizza without a scooper after entering a predefined Region of Interest (ROI).
    *   **Input:** Detected objects and frames from RabbitMQ.
    *   **Logic:**
        *   Tracks the state of each hand (entered ROI, exited ROI, touched pizza, had scooper).
        *   Identifies violations based on a sequence of events (hand enters ROI -> hand exits ROI -> hand touches pizza -> no scooper present).
        *   Saves violation details to a SQLite database (`violations.db`).
    *   **Output:** Publishes annotated frames (with bounding boxes, ROI, and violation count) to RabbitMQ.

3.  **`streaming-service`**:
    *   **Purpose:** Acts as the front-end for the system, consuming annotated frames and violation counts from the `violation-service`.
    *   **Input:** Annotated frames and violation counts from RabbitMQ.
    *   **Output:** Provides a live video stream and a dashboard via a Flask web application, displaying the real-time annotated video and the total number of violations.

**Communication Flow:**

Video Input -> `detection-service` (YOLO detection) -> RabbitMQ (`detected_objects` queue) -> `violation-service` (Violation logic, Annotation) -> RabbitMQ (`annotated_frames` queue) -> `streaming-service` (Web Dashboard)

## Setup and Installation

To get the project up and running, ensure you have Docker and Docker Compose installed on your system.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd PizzaStoreScooperViolation-Detection-
    ```

2.  **Build and run the Docker containers:**
    ```bash
    docker-compose up --build
    ```
    This command will:
    *   Build the Docker images for `detection-service`, `violation-service`, and `streaming-service`.
    *   Start all services, including a RabbitMQ instance.

3.  **Access the application:**
    Once all services are running, open your web browser and navigate to:
    ```
    http://localhost:8000
    ```
    You should see the live video stream with object detections and the violation count.

## Usage

*   The system will automatically start processing the video file specified within the `detection-service` (currently `/dataset/3.mp4` as configured in `detection-service/app.py`).
*   The web dashboard at `http://localhost:8000` will display the annotated video stream.
*   Violations will be highlighted on the stream, and the total violation count will be updated in real-time.
*   Violation details are stored in `violations.db` within the `violation-service` container.

## Project Structure

```
PizzaStoreScooperViolation-Detection-/
├───docker-compose.yml              # Defines and orchestrates the multi-container Docker application
├───README.md                       # This documentation file
└───microservices/
    ├───detection-service/          # Service for object detection and tracking
    │   ├───app.py                  # Main application logic for detection
    │   ├───Dockerfile              # Dockerfile for building the detection service image
    │   ├───requirements.txt        # Python dependencies for detection service
    │   ├───dataset/                # Contains video files for processing
    │   │   ├───1.mp4
    │   │   ├───2.mp4
    │   │   └───3.mp4
    │   └───models/                 # Pre-trained YOLO models
    │       ├───best.pt
    │       └───yolo12/
    │           ├───best.pt
    │           └───yolo12m-v2.pt
    ├───streaming-service/          # Service for streaming annotated video and dashboard
    │   ├───app.py                  # Main application logic for streaming
    │   ├───Dockerfile              # Dockerfile for building the streaming service image
    │   ├───requirements.txt        # Python dependencies for streaming service
    │   └───templates/              # HTML templates for the web dashboard
    │       └───index.html
    └───violation-service/          # Service for applying violation logic and logging
        ├───app.py                  # Main application logic for violation detection
        ├───Dockerfile              # Dockerfile for building the violation service image
        ├───requirements.txt        # Python dependencies for violation service
        ├───violations.db           # SQLite database for storing violation records
        └───DetectionAndViolation/  # Module containing violation-related utilities
            ├───violation_database.py # Database interaction logic
            └───__pycache__/        # Python cache directory
```
