# Hand Control Desktop App

A desktop application that uses hand tracking to control system brightness, volume, and to take screenshots using hand gestures. The app leverages PyQt5 for the GUI and integrates OpenCV, TensorFlow, and other libraries to provide gesture-based controls along with global hotkeys that work system-wide.

## Features

- **Hand Gesture Controls:**
  - **Brightness Mode:** Adjust screen brightness with hand gestures.(working for windows only)
  - **Volume Mode:** Adjust system volume with hand gestures.
  - **Screenshot Mode:** Take a screenshot using an open-hand gesture.
- **Global Hotkeys:**
  - **Mode Switching:**
    - `Ctrl+9` – Switch to Brightness Mode.
    - `Ctrl+8` – Switch to Volume Mode.
    - `Ctrl+7` – Switch to Screenshot Mode.
  - **Lock Toggling:**
    - `Ctrl+Shift+B` – Toggle Brightness Lock (prevents hand gesture changes).
    - `Ctrl+Shift+V` – Toggle Volume Lock (prevents hand gesture changes).

*Global hotkeys work even if the application window is not in focus, so you can control your system settings while working in other applications (e.g., YouTube in a browser).*

## Requirements

- **Python 3.7+**
- **Libraries:**
  - [PyQt5](https://pypi.org/project/PyQt5/)
  - [opencv-python](https://pypi.org/project/opencv-python/)
  - [NumPy](https://pypi.org/project/numpy/)
  - [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)
  - [screen_brightness_control](https://pypi.org/project/screen-brightness-control/)
  - [TensorFlow](https://www.tensorflow.org/)
  - [pynput](https://pypi.org/project/pynput/)
  - [pycaw](https://pypi.org/project/pycaw/) *(for Windows volume control)*
- **Custom Module:**
  - `handtrackingmodule.py` (Ensure this file is in your project directory.)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/hand-control-desktop-app.git
   cd hand-control-desktop-app
   ```


2. **Create a virtual environment (optional but recommended):**

```bash
python3 -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```

3. **Install the required packages:**

Create a requirements.txt file (if not already provided) with content similar to:

```bash
nginx
Copy
Edit
PyQt5
opencv-python
numpy
pyautogui
screen_brightness_control
tensorflow
pynput
pycaw  # Only needed for Windows
```
Then install using pip:

```bash
pip install -r requirements.txt
```

**Ensure handtrackingmodule.py is present in the same directory as your main script (e.g., app.py).**

**Usage**
To run the application, execute the main Python script:

```bash
python app.py
```
- **Upon Launch:**
  - The app opens a window displaying the webcam feed.
  - An overlay shows the current mode and instructions.

- **Operating Modes:**
  - **Main Menu:**  
    - Default mode; instructs you to use global hotkeys.
  - **Brightness Mode:**  
    - Adjust system brightness with hand gestures.
  - **Volume Mode:**  
    - Adjust system volume with hand gestures.
  - **Screenshot Mode:**  
    - Take screenshots via an open-hand gesture (subject to a cooldown period).

- **Global Hotkeys (Available Regardless of Active Window):**
  - **Mode Switching:**
    - **Ctrl+9** – Switch to Brightness Mode.
    - **Ctrl+8** – Switch to Volume Mode.
    - **Ctrl+7** – Switch to Screenshot Mode.
  - **Lock Toggling:**
    - **Ctrl+Shift+B** – Toggle Brightness Lock.
    - **Ctrl+Shift+V** – Toggle Volume Lock.
  - **Note:**  
    - Ensure these key combinations do not conflict with other system shortcuts.









