# Blood Vessel Detection System

This project implements various image processing techniques for blood vessel detection in medical images, including Adaptive Principal Curvature (APC) method, Canny edge detection, and other edge detection algorithms.

## Features

- Multiple edge detection methods:
  - Adaptive Principal Curvature (APC)
  - Canny Edge Detection
  - Central Difference
  - Various classical edge detection filters (Sobel, Prewitt, Roberts, Scharr)
- GUI interface using tkinter (when available)
- Image visualization using matplotlib
- Pre-processing and post-processing capabilities

## Prerequisites

Before running this project, make sure you have Python 3.7+ installed on your system.

## Installation

1. Clone this repository or download the project files:
```bash
git clone <repository-url>
cd "APC Blood Vessal Block 3"
```

2. Install the required packages using pip:
```bash
pip install numpy opencv-python scipy scikit-image matplotlib
```



## Usage

To run the application:

```bash
python app.py
```

If tkinter is available, the program will launch with a graphical user interface. Otherwise, it will run in command-line mode.

## Dependencies

The following Python packages are required:

- numpy
- opencv-python (cv2)
- scipy
- scikit-image
- matplotlib (optional, for visualization)
- tkinter (optional, for GUI)

## Error Handling

The program includes error handling for:
- Missing tkinter installation (falls back to command-line mode)
- Missing matplotlib installation (falls back to OpenCV for display)
- Image loading and processing errors

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Specify your license here]
