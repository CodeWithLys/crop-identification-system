READ.ME

📂Important Notes on Paths
The YOLOv5 scripts are very sensitive to file paths. For the model to work correctly, all paths in commands must be adjusted to match the exact location where the folders and files are saved on your computer.

If the paths do not match, the model will fail to find weights or data files and will throw errors.


⚠️How to Run Detection Using Command Line

##Open your terminal or command prompt.

1. Navigate to the yolov5-master directory. For example:
##Run the following command from inside the yolov5-master directory:

cd "C:\Users\Gabriel Krishna\Desktop\Alyssa\IFS 315\Vision-based System\Final Trained Model\yolov5-master"


2. Your dataset.yaml file is correctly located in the yolov5-master folder and contains the correct paths like:

train: ../raw_data/data/dataset/images/train
val: ../raw_data/data/dataset/images/val
nc: 11
names: ['apple', 'banana', 'carrot', 'corn', 'grapes', 'kiwi', 'lettuce', 'onion', 'pineapple', 'potato', 'tomato']
 
## You have YOLOv5 dependencies installed (including PyTorch and OpenCV).

## yolov5s.pt is present in the same directory (yolov5-master) or accessible via the internet to auto-download.

3. How to Train the Model Using Command Line

python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --name exp6

##Make sure you run the training from inside the yolov5-master folder with the correct data yaml and weights path


‼️🎥REAL-TIME WEBCAM DETECTION

python detect.py --weights runs/train/exp6/weights/best.pt --img 640 --conf 0.25 --source 0


Best model file:

runs/train/exp6/weights/best.pt