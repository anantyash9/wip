import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Cropping2D, MaxPooling2D, Dropout, Reshape, \
    Convolution2D
from moviepy.editor import VideoFileClip
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


class FishingDetection:
    def __init__(self, human_detection_graph_path,
                 human_detection_graph_label_path,
                 num_classes,
                 fishing_classification_graph_path,
                 fishing_classification_graph_label_path):
        self.__human_detection_graph_path = human_detection_graph_path
        self.__human_detection_graph = FishingDetection.load_graph(self.__human_detection_graph_path)
        self.__num_classes = num_classes
        self.__fishing_classification_graph_path = fishing_classification_graph_path
        self.__fishing_classification_graph = FishingDetection.load_inception(self.__fishing_classification_graph_path)
        self.__label_map = label_map_util.load_labelmap(human_detection_graph_label_path)
        self.__categories = label_map_util.convert_label_map_to_categories(self.__label_map,
                                                                           max_num_classes=self.__num_classes,
                                                                           use_display_name=True)
        self.__category_index = label_map_util.create_category_index(self.__categories)

    def nvidia_net(self, drop_prob=0.2):
        # create a sequential Model
        model = Sequential()
        # Add a Cropping layer to trim the unneeded portions of the IMAGE from the feed
        # model.add(Cropping2D)
        # model.add(Reshape((50,50,3), input_shape=(None,None,3))
        model.add(Cropping2D(cropping=((0, 0), (0, 0)), input_shape=(100, 100, 3)))
        # Normalization Layer
        # model.add(Lambda(lambda X_input: (X_input/255.0 - 0.5)))
        # Conv2D Layer 1 with 5 x 5 kernal size
        # model.add(Convolution2D(nb_filter = 3, nb_row = 5, nb_col = 5))
        model.add(Convolution2D(nb_filter=3, nb_row=3, nb_col=3, subsample=(1, 1)))
        model.add(Activation('relu'))
        # Dropout layer
        model.add(Dropout(drop_prob))
        # Conv2D Layer 2 with 5 x 5 kernal size
        # model.add(Convolution2D(nb_filter = 24, nb_row = 5, nb_col = 5))
        model.add(Conv2D(nb_filter=24, nb_row=3, nb_col=3, subsample=(2, 2)))
        model.add(Activation('relu'))
        # Conv2D Layer 3 with 5 x 5 kernal size
        # model.add(Convolution2D(nb_filter = 36, nb_row = 5, nb_col =  5))
        model.add(Conv2D(nb_filter=36, nb_row=3, nb_col=3, subsample=(1, 1)))
        model.add(Activation('relu'))
        # Dropout layer
        model.add(Dropout(drop_prob))
        # Conv2D Layer 4 with 3 x 3 kernal size
        # model.add(Convolution2D(nb_filter = 48, nb_row = 3, nb_col = 3))
        model.add(Conv2D(nb_filter=48, nb_row=3, nb_col=3))
        model.add(Activation('relu'))
        # Conv2D Layer 5 with 3 x 3 kernal size
        # model.add(Convolution2D(nb_filter = 48, nb_row = 3, nb_col = 3))
        model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3))
        model.add(Activation('relu'))
        # flatten layer
        # flatten the output from convolution Layer
        model.add(Flatten())
        # Fully connected layer 1
        model.add(Dense(output_dim=50))
        # Fully connected Layer 2
        model.add(Dense(output_dim=25))
        # Fully connected Layer 3
        model.add(Dense(output_dim=10))
        # output Layers
        model.add(Dense(output_dim=2, activation='softmax'))
        return model

    @staticmethod
    def load_graph(graph_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    @staticmethod
    def load_inception(graph_path):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self,
                                       image,
                                       graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def list_files(self, path):
        # files = [f for f in listdir(path) if isfile(join(path, f))]
        # return files
        return glob.glob(path)

    def process_files(self,
                      list_files,
                      label):
        output = []
        for i in list_files:
            temp = []
            temp.append(i)
            ground_truth = [0, 0]
            ground_truth[label] = 1.
            temp.append(ground_truth)
            output.append(temp)
        return output

    def read_image(self, path):
        img = Image.open(path).convert('RGB')
        img = self.make_square(img)
        img = img.resize((100, 100), Image.ANTIALIAS)
        return np.asarray(img)

    def pred_keras(self,
                   img,
                   keras_model):
        # print(img)
        img = Image.fromarray(img)
        img = self.make_square(img)
        img = img.resize((100, 100), Image.ANTIALIAS)
        return keras_model.predict(np.expand_dims(np.asarray(img), axis=0))

    def make_square(self,
                    im,
                    min_size=100,
                    fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        # print((int((size - x) / 2), int(size - y) / 2))
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    # prediction step
    def load_trained_model_keras(self, weights_path):
        model = self.nvidia_net()
        model.load_weights(weights_path)
        return model

    def load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def read_tensor_from_image_file(self,
                                    image_np,
                                    input_height=299,
                                    input_width=299,
                                    input_mean=0,
                                    input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = image_np
        image_reader = tf.convert_to_tensor(image_np, np.uint8)
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)
        return result

    def pred_tensorflow(self,
                        image_np,
                        graph,
                        input_height=299,
                        input_width=299,
                        input_mean=0,
                        input_std=255):
        sess = tf.Session(graph=graph)
        # label_file = r'D:\fishing_detection\data\output_labels.txt'
        t = self.read_tensor_from_image_file(
            image_np,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)
        input_name = "import/" + "Placeholder"
        output_name = "import/" + "final_result"
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)
        with sess as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)
        return results

    def predict_fishing(self, image, boxes, classes, scores):
        im_width, im_height = image.size
        image_np = self.load_image_into_numpy_array(image)
        predictions = {}
        predictions['bo'] = []
        predictions['res'] = []
        predictions['score'] = []
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > 0.5:
                predictions['bo'].append(boxes[i])
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)
                roi = image_np[ymin:ymax, xmin:xmax]
                # c = pred(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                c = self.pred_tensorflow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), self.__fishing_classification_graph)
                if (c[0] > 0.5):
                    predictions['res'].append(0)
                    predictions['score'].append(c[0])
                else:
                    predictions['res'].append(1)
                    predictions['score'].append(c[1])

        return predictions

    def pipeline(self, img):
        image = Image.fromarray(img)
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = self.run_inference_for_single_image(image_np, self.__human_detection_graph)
        for i in range(len(output_dict['detection_classes'])):
            if output_dict['detection_classes'][i] != 1:
                output_dict['detection_scores'][i] = 0.0
        predi = self.predict_fishing(image, output_dict['detection_boxes'], output_dict['detection_classes'],
                                     output_dict['detection_scores'])
        category_ind = {1: {'id': 1, 'name': 'NOT FISHING'}, 0: {'id': 0, 'name': 'FISHING'}}
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.asarray(predi['bo'], dtype=np.float32),
            predi['res'],
            predi['score'],
            category_ind,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np


def main():
    INPUT_DIRECTORY = 'input_video'
    OUTPUT_DIRECTORY = 'output_video'
    INPUT_FILE = 'Catching_tuna_Maldivian_style.mp4'
    OUTPUT_FILE = 'result4.mp4'
    vid_output = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)
    vid_input = os.path.join(INPUT_DIRECTORY, INPUT_FILE)
    subclip_start = '00:01:15.50'
    subclip_end = '00:01:15.80'
    clip = VideoFileClip(vid_input).subclip(subclip_start, subclip_end)

    DIRECTORY_NAME = 'models'
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    TAR_EXTENSION = '.tar.gz'
    MODEL_FILE = MODEL_NAME + TAR_EXTENSION
    GRAPH_NAME = 'frozen_inference_graph.pb'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_FROZEN_GRAPH = os.path.join(DIRECTORY_NAME, MODEL_NAME, GRAPH_NAME)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90
    fishing_detection_graph = os.path.join('models', 'inception_retrained', 'output_graph.pb')
    fishing_classification_graph_label_path = os.path.join('data', 'output_labels.txt')
    object_detect = FishingDetection(human_detection_graph_path=PATH_TO_FROZEN_GRAPH,
                                     human_detection_graph_label_path=PATH_TO_LABELS,
                                     num_classes=NUM_CLASSES,
                                     fishing_classification_graph_path=fishing_detection_graph,
                                     fishing_classification_graph_label_path=fishing_classification_graph_label_path)
    vid = clip.fl_image(object_detect.pipeline)
    vid.write_videofile(vid_output, audio=False)


if __name__ == '__main__':
    main()
