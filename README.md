# TAPS: Tactile-Acoustic Perception for Vision-denied Robot Operation
- Final team project for *EECE490G: Robot Learning @POSTECH*
- See our [paper](/figure/paper.pdf) for more details!

![project](/figure/project.png)

---
## Overview

**When robots can't see, they can still feel and listen.**

TAPS is a vision-denied robotic perception system that recognizes objects through **active touch and sound**. By combining contact-based geometric exploration with impact-sound-based material classification, TAPS enables reliable object recognition without cameras or LiDAR, making it suitable for low-visibility environments such as smoke, darkness, or occlusion.

---
## Demo

https://github.com/user-attachments/assets/ab322c13-d82c-4b98-85c9-57fcaeeaff32

---
## Usage

1. **Set up the environment:**
   ```sh
   conda create -n TAPS python=3.11
   conda activate TAPS
   pip install -r requirements.txt
   ```
2. **Download pre-trained models:**
   - [Google Drive](#) *(link to be added)*
   - Place the downloaded model files in the `models/` directory.

3. **Run the main control script:**
   ```sh
   python move/ver3.0/continuous_move.py
   ```

4. **Keyboard Controls:**
   - `W`, `A`, `S`, `D`: Move the robot base (manual control)
   - `G`: X-axis (width) exploration — finds and memorizes the object's center
   - `Z`: Z-axis (height) exploration — measures object height at the memorized center
   - `Q`: Vertical collision, records impact sound, and classifies the object
   - `X`: Emergency stop/exit
   - `R`: Open gripper
   - `T`: Close gripper
   - `E`, `C`, `U`, `J`, `I`, `K`: Fine joint controls (see code for details)

5. **Workflow Example:**
   1. Press `G` to perform X-axis search and memorize the object's center.
   2. Press `Z` to perform Z-axis search at the memorized center and measure height.
   3. Press `Q` to strike the object, record the impact sound, and classify the object.
---
## Adding Custom Objects

To add your own objects for recognition, edit the `object_database.yaml` file as follows:

1. Open `move/ver3.0/object_database.yaml` in a text editor.
2. Under the `objects:` section, add a new entry with the following fields:
   - `id`: A unique identifier for your object (e.g., "obj_07")
   - `name`: The object's name (e.g., "Water Bottle")
   - `material`: The object's material (e.g., "plastic", "aluminum", "paper")
   - `radius`: The object's radius in centimeters (cm)
   - `height`: The object's height in centimeters (cm)

3. Save the file. The system will now be able to recognize your custom object based on its estimated size and material (classification into the nearest object).
---
## Acknowledgement
This codebase is built upon [Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot) and [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn).