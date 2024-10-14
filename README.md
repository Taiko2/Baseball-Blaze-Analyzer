# BlazePose Baseball Analyzer

This repository contains a program that uses Mediapipe BlazePose to analyze and track baseball player positions in a video. The system processes video frames to track key body landmarks, excluding face and hand tracking, focusing on the body posture, feet, and a singular wrist point.

## Requirements

To run this program, you need the following dependencies installed:

- mediapipe==0.10.0
- opencv-python==4.5.5.64
- numpy==1.22.0
- tensorflow==2.10.0

You can install these using the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/Taiko2/Baseball-Blaze-Analyzer.git
cd BlazePose-Baseball-Analyzer
```

2. Install the dependencies:

```sh
pip install -r requirements.txt
```

3. Place your input video in the designated input folder or specify the path in the code. The default input video path is:

```
C:\Users\xxx\Desktop\Python body tracking\ACslowmo.mp4
```

4. Run the program using Python:

```sh
python main.py
```

5. The output video with the analyzed tracking will be saved to:

```
C:\Users\xxx\Desktop\Python body tracking\output\output.mp4
```

## Features

- Tracks key body landmarks, focusing on the major joints and avoiding face and finger tracking.
- Visualizes pose analysis using colored circles for keypoints and lines for body connections.
- Provides a simplified visualization for easier analysis of baseball player positioning.
