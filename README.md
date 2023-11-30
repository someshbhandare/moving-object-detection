<!-- # moving-object-detection
Detect and classify the moving objects in a video using YOLO and Optical Flow Estimation -->

# Moving Objects Detection & Classification

### Object Detection and Classification using YOLO and Optical Flow Estimation

<!-- ![Project Image/Logo](path/to/image.png) -->

<!-- ## Description -->

This project utilizes YOLO (You Only Look Once) for object detection and classification in videos, complemented by optical flow estimation for improved tracking of moving objects.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/someshbhandare/moving-object-detection
    cd moving-object-detection
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the main script:**

    ```bash
    python main.py --input_video path/to/your/video.mp4 --output_video_path output.mp4
    ```

    Replace `path/to/your/video.mp4` with the path to your input video file.
    
    - --input_video_path = YOUR_INPUT_VIDEO_PATH
    - --output_video_path = YOUR_OUTPUT_VIDEO_PATH [optional]


## Results

\#

## Contributing

If you would like to contribute to the project, follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the [Apache License 2.0](LICENSE).

