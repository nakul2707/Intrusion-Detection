<!-- Intrusion Detection with YOLOv5 and Face Recognition -->

<!-- Overview -->
This project implements an intrusion detection system that combines YOLOv5 for person detection with the face_recognition library for face verification. A user-defined tripwire line is drawn on the input video. When a person crosses this line:
•	If their face matches one of the authorized faces, they are labeled AUTHORIZED (green box).
•	Otherwise, they are labeled INTRUDER (red box) and a siren sound is played.

<!-- Repository Structure -->
├── index.py           # Main script for intrusion detection
├── Faces/             # Directory of authorized face images
│   ├── person1D_face0.jpg
│   └── person2D_face0.jpg
├── siren.wav          # Siren sound file for intruder alert
├── yolov5n.pt         # Pre-trained YOLOv5 model weights
├── requirements.txt   # Python dependencies
├── utils/             # YOLOv5 utility modules (augmentation, NMS, etc.)
├── models/            # YOLOv5 model definitions and backends
├── data/              # Data loaders and sample datasets
├── detect.py          # YOLOv5 detection demo script
├── train.py           # YOLOv5 training script
└── README.md          # This file

<!-- Prerequisites -->
•	Python 3.7+
•	ffmpeg (optional, for video formats beyond OpenCV defaults)

Install Python dependencies:

pip install -r requirements.txt

The key Python packages include:
•	torch, torchvision (PyTorch)
•	opencv-python (OpenCV bindings)
•	face_recognition (face detection & encoding)
•	pygame (audio playback)
•	numpy, argparse

<!-- Usage -->
1.	Install Dependencies
Before running the script, install all required packages with:
pip install -r requirements.txt
2.	Prepare Authorized Faces
o	Place clear face images of authorized individuals in the Faces/ folder.
o	Filenames should be unique (e.g., person1.jpg, person2.jpg).
3.	Run the Script
4.	python index.py \
5.	  --video Input.mp4 \
6.	  --output Output.mp4 \
7.	  --weights yolov5n.pt \
8.	  --known-dir Faces \
9.	  --siren siren.wav \
10.	  --logo logo.jpg \
11.	  --shape 320 \
12.	  --dist 30.0 \
13.	  --tol 0.6 \
14.	  --cooldown 5.0
o	--video: Path to the input video file.
o	--output: Path to save the annotated output video.
o	--weights: YOLOv5 model weights.
o	--known-dir: Directory with authorized face images.
o	--siren: Siren sound file to play on intrusion.
o	--logo: (Optional) PNG logo overlay file.
o	--shape: Size for YOLOv5 image resizing (shorter side).
o	--dist: Distance threshold (pixels) to tripwire for intrusion.
o	--tol: Face match tolerance (lower = stricter).
o	--cooldown: Cooldown time (seconds) between siren blasts.
15.	Define the Tripwire
o	When prompted, two points will appear on a window.
o	Click two points on the video frame to set the tripwire.
o	Press q or Esc to abort.
16.	View Results
o	A window titled Output shows real-time annotations.
o	The processed video is saved to the specified --output path.
How It Works
•	Tripwire Definition: Users click two points on the first frame to define a line.
•	Person Detection: YOLOv5 (DetectMultiBackend) detects all persons in each frame.
•	Face Recognition: For each detected person, the face area is cropped and encoded. Encodings are compared to known faces.
•	Intrusion Logic:
1.	Compute the distance from a person’s face centroid to the tripwire line.
2.	If distance ≤ --dist and shape crossing is detected:
	Match (distance to known encodings ≤ --tol): label AUTHORIZED.
	No Match: label INTRUDER, play siren (if not in cooldown).
Logging & Output
•	The script prints debug/info messages to the console.
•	Final output video with annotations is saved to --output.
•	Audio (siren) plays via pygame when an intrusion is detected.

<!-- ________________________________________ -->
<!-- Note: -->
1.	First, open the index.py file.
2.	Ensure that the faces in the Faces/ folder are correctly labeled and recognized by the model.
3.	For testing, place two people’s face images (person1, person2) in the Faces/ folder.
4.	When running the video, any face not matching these images will be flagged as an intruder and the siren will sound.
5.	To start the system, run in the terminal:

python index.py

