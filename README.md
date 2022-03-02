# OpenCV Object Detection in Games
I made this project specifically to apply some of what I've been learning with opencv to games. Initially I am looking to train a haar classifier to identify targets in the game Aim Labs. I chose this game because the targets are relatively static, easy to see,  and it should be easy to gerate positive and negative images and annotate them all. The classifier was trained using opencv's 'opencv_traincascade' application with images annotated with 'opencv_annotation'. 
# Initial results:
![python_6KJ1cVgTW1](https://user-images.githubusercontent.com/47011094/156307620-79a484eb-d7ca-4977-8524-4600f08e271b.png)
The first iteration was successfully able to identify each target, but is also giving a lot of false positives. This is to be expected when I used a very low positive and negative sample size.
