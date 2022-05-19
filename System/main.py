import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn

import pytesseract
import re


@dataclass
class Configuration:
    cities: tuple = ('Birmingham', 'Chester', 'Dublin', 'Edinburgh', 
                     'Exeter', 'Glasgow', 'London', 'Newcastle', 'Sheffield')
    letters: tuple = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
    russian: tuple = ('1896-1908', 'before1896', 'from1908')
    

    hallmark_image_shape: tuple = (70, 70)
    letter_image_shape: tuple = (70, 70)
    russian_image_shape: tuple = (100, 100)

    path_to_letter_model_weights: str = 'configuration/letter_classification.pth'
    path_to_city_model_weights: str = 'configuration/city_classification.pth'
    path_to_russian_model_weights: str = 'configuration/russian_classification.pth'

    path_to_detection_model_weights: str = 'configuration/yolov4-obj.weights'
    path_to_detection_config_file: str = 'configuration/yolov4-obj.cfg'

    # Name of the classes for detection
    with open("configuration/classes.txt") as f:
        class_names = [line.strip() for line in f]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def init_resnet_model(config: dataclass, path_to_weights: str) -> models.ResNet:
    """
    The function return object of model with uploaded weights for certain task of classification


    :param config: dataclass object with all necessary information
    :param path_to_weights: path to .pth file with weights of model
    :return:
    """

    device = config.device
    model_state = torch.load(path_to_weights, map_location=torch.device(device))
    output_number_of_classes, input_number_of_features = model_state['fc.weight'].shape

    # Model architecture initialization
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(input_number_of_features, output_number_of_classes)

    # Loading state of the model
    model.load_state_dict(model_state)

    # Transferring the model to the device and freezing the weights
    model.to(device)
    model.eval()

    return model


def init_squeezenet_model(config: dataclass, path_to_weights: str) -> models.SqueezeNet:
    """
    The function return object of model with uploaded weights for certain task of classification


    :param config: dataclass object with all necessary information
    :param path_to_weights: path to .pth file with weights of model
    :return:
    """

    device = config.device
    model_state = torch.load(path_to_weights, map_location=torch.device(device))
#     output_number_of_classes, input_number_of_features = model_state['fc.weight'].shape
    output_number_of_classes, input_number_of_features = model_state.classifier[1].weight.shape[0], model_state.classifier[1].weight.shape[1]

    # Model architecture initialization
    model = models.squeezenet1_1(pretrained=False)
    model.classifier[1] = nn.Conv2d(input_number_of_features, output_number_of_classes, kernel_size=(1,1), stride=(1,1))

    # Loading state of the model
    model.load_state_dict(model_state.state_dict())

    # Transferring the model to the device and freezing the weights
    model.to(device)
    model.eval()

    return model


def init_detection_model(config: dataclass) -> cv2.dnn_DetectionModel:
    """
    The function return object of model with uploaded weights for detection task
    :param config: dataclass object with all necessary information
    :return:
    """
    path_to_weights = config.path_to_detection_model_weights
    path_to_config = config.path_to_detection_config_file

    net = cv2.dnn.readNet(path_to_weights, path_to_config)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    return model


def image_to_gray(img):
    gray = cv2.resize(img, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(gray, rect_kern, iterations = 1)

    return dilation

def predict_year(im):
    gray =  cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    if h/w > 1: 
        gray = cv2.rotate(gray, cv2.cv2.ROTATE_90_CLOCKWISE)
        im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
    
    gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    im = cv2.resize(im, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray, 3)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    # find contours
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # create copy of image
    im2 = dilation.copy()
    height, width = im2.shape

    year = ""

    color = (0,215,255)
    # loop through contours and find letters in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if height / float(h) > 3: continue
        ratio = h / float(w)
        if ratio < 1.25: continue
        area = h * w
        if width / float(w) > 20: continue
        if area < 100: continue
            
        # draw the rectangle
        rect = cv2.rectangle(im, (x,y), (x+w, y+h), color,2)
        padding = 0
        roi = thresh[y-padding:y+h+padding, x-padding:x+w+padding]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)

        text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3')
        clean_text = re.sub('[\W_]+', '', text)
        year += clean_text
        
    return year

class HallmarkAnalyser:
    def __init__(self, config: dataclass):
        """

        :param config: class with all configuration parameters
        """
        self.config = config
        self.device = config.device

        # Initialization of all necessary models
        self.letter_model = init_resnet_model(config, config.path_to_letter_model_weights)
        self.city_model = init_squeezenet_model(config, config.path_to_city_model_weights)
        self.russian_model = init_squeezenet_model(config, config.path_to_russian_model_weights)
        self.detection_model = init_detection_model(config)

        # Detection parameters
        self.detection_threshold = 0.2
        self.nms_threshold = 0.4
        self.colors = ((0,241,255), (0,114,255), (255, 0,241), (14,0,255), (255,0,114))

    def make_prediction(self, image: np.ndarray, type_of_task: str) -> tuple:
        """
        Function provide classification of incoming image
        :param image: Cropped image with hallmark or letter
        :param type_of_task: type of classification task ('hallmark' or 'letter')
        :return: tuple with predicted label and confidence of prediction
        """

        if type_of_task == 'Year':
            start = time.time()
            predicted_label = predict_year(image)
            end = time.time()
            print(f'Classification {type_of_task} inference time : {(end - start):.2f} seconds')
            return predicted_label, 0
        
        if type_of_task == 'Letter':
            model = self.letter_model
            image_shape = self.config.letter_image_shape
            labels = self.config.letters
        elif type_of_task == 'Hallmark':
            model = self.city_model
            image_shape = self.config.hallmark_image_shape
            labels = self.config.cities
        elif type_of_task == 'Russian':
            model = self.russian_model
            image_shape = self.config.hallmark_image_shape
            labels = self.config.russian
               
                        
        print(type_of_task)
        image = image_to_gray(image)
        image = cv2.resize(image, image_shape)
        image = torch.tensor(image / 255).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        start = time.time()
        prediction = model(image).squeeze().detach().cpu()
        end = time.time()

        print(f'Classification {type_of_task} inference time : {(end - start):.2f} seconds')

        predicted_class = prediction.argmax()
        predicted_label = labels[predicted_class]
        confidence = prediction.sigmoid()[predicted_class].item()

        return predicted_label, confidence

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Function process image of silverware. (detects the hallmark and classify the town and letter)
        :param image: image for hallmark analysis
        :return: dictionary with keys Detection classes and values tuple of classification classes
        """
        image = cv2.resize(image, (416, 416))
        
        start = time.time()
        classes, scores, boxes = self.detection_model.detect(image, self.detection_threshold, self.nms_threshold)
        end = time.time()

        print(f'Detection inference time : {(end - start):.2f} seconds')

        analysis_results = {}
        print('classes', classes, 'len', len(classes))
        for (class_id, score, box) in zip(classes, scores, boxes):
            color = self.colors[int(class_id)]

            x_left, y_top, x_right, y_bottom = box[0], box[1], box[0] + box[2], box[1] + box[3]
            image_hallmark = image[y_top:y_bottom, x_left:x_right]

            print('class_id', class_id)
            if class_id == 0:
                label = f"{self.config.class_names[class_id]}: {float(score):.2f}"
                text_position = (box[0]-5, box[1] - 5)
                cv2.rectangle(image, box, color, 1)
                cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                return image, analysis_results
            
            
            classification_label, confidence = self.make_prediction(image_hallmark,
                                                                    self.config.class_names[
                                                                        class_id])
            analysis_results[self.config.class_names[class_id]] = (classification_label, confidence)

            if class_id == 4:
                label = f'{self.config.class_names[class_id]}: {float(score):.2f}, {classification_label}'
            else:
                label = f'{self.config.class_names[class_id]}: {float(score):.2f}, {classification_label}: {confidence:.2f}'

            if class_id == 2 or class_id == 4:
                text_position = (box[0], box[1] + box[2] + 10) 
            else:
                text_position = (box[0] - 5, box[1] - 5)

            cv2.rectangle(image, box, color, 1)
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image, analysis_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--image', type=str, required=False, default='TestImages/0.jpg', help='Path to image')

    args = parser.parse_args()
    image_path = args.image

    cfg = Configuration()
    analyser = HallmarkAnalyser(cfg)

    image_input = cv2.imread(image_path)
    assert image_input is not None, "Image not found"

    # Change BGR to RGB
    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    image_output, results = analyser.process_image(image_input)

    # Print results of models
    for key in results.keys():
        print(f"{key} : {results[key][0]} - {results[key][1]}")

    try:
        url = "https://silvermakersmarks.co.uk/Dates/{}/Date%20Letters%20{}.html".format(
            results['Hallmark'][0], results['Letter'][0])
        print(url)
    except:
        pass
        
    plt.figure(figsize=(10, 10))
    plt.imshow(image_output)
    plt.show()
#     #######################################################################################
#     cfg = Configuration()
#     analyser = HallmarkAnalyser(cfg)

#     image_input = cv2.imread(image_path)
#     assert image_input is not None, "Image not found"

#     # Change BGR to RGB
#     image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
#     image_output, results = analyser.process_image(image_input)

#     # Print results of models
#     f = open(f"{save_path}.txt", "w")
#     for key in results.keys():
#         f.write(f"{key} : {results[key][0]} - {results[key][1]}" + "\n")
#         print(f"{key} : {results[key][0]} - {results[key][1]}")

#     try:
#         url = "https://silvermakersmarks.co.uk/Dates/{}/Date%20Letters%20{}.html".format(
#             results['Hallmark'][0], results['Letter'][0])
#         f.write(url)
#         print(url)
#     except:
#         pass
#     f.close()
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_output)
#     plt.savefig(save_path)
#     plt.show()

