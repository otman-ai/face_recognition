# Face Recognition System
A [facial recognition system](https://en.wikipedia.org/wiki/Facial_recognition_system) is a technology capable of matching a human face from a digital image or a video frame against a database of faces. Such a system is typically employed to authenticate users through ID verification services, and works by pinpointing and measuring facial features from a given image.
## Exaplain the contents of the repository:
This Repository you will find:
* `data_registred` which is a folder that track all the faces that the model detect to be known
* `Images` a folder that content all the images that you want the model to know them.
* * `Note`:the image name should be the name of person in the image.
* `Knowingfaces.py`: python scrpits that read the image from the folder `Images` and make encode the face it detect on all the images with and then save it  as csv format with their labels in `face_encoding.csv`.
* `app.py`:python scripts that run the webcam and detect the faces their are already in encoded in `face_encoding.csv`

## How to run the repository?

To use this repo first, you need to clone it with this cammand below:

```
git clone https://github.com/otman-ai/face_recognition.git
```
Install all the requirements:
```
pip install -r requirements.txt
```
Put the images you want to detect inside the Images folder;make sure the name of the image is correspond to the name of the person in the image.
Then run the following command
```
python app.py
```