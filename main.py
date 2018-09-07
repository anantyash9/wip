
# coding: utf-8

# # Imports

# In[5]:


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
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Cropping2D, MaxPooling2D, Dropout, Reshape,     Convolution2D
from moviepy.editor import VideoFileClip
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
import json


# ## Visualization utils

# In[ ]:


import collections
import functools

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen']
def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,classes1,
    scores,scores1,
    category_index,category_index1,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
              class_name1 = category_index1[classes1[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(class_name, int(100*scores[i]))
            display_str +=' :: '
            display_str += '{}: {}%'.format(class_name1, int(100*scores1[i]))
        box_to_display_str_map[box].append(display_str)
        
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      vis_util.draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      vis_util.draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    vis_util.draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      vis_util.draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image

def make_frame_log(
    boxes,
    classes,classes1,
    category_index,category_index1):
  fishermen ={}
  for i in range(boxes.shape[0]):
    class_name = category_index[classes[i]]['name']
    class_name1 = category_index1[classes1[i]]['name']
    fishermen[i] =[class_name,class_name1]
  return fishermen  

def make_dummer_log(dict_log):
  dumb_dict = {}
  dumb_dict['people']=[]
  dumb_dict['length']=[]
  dumb_dict['fishing']=[]
  dumb_dict['not_fishing']=[]
  dumb_dict['geared']=[]
  dumb_dict['not_geared']=[]
  dumb_dict['fishing_and_not_geared']=[]
  dumb_dict['fishing_and_geared']=[]
  dumb_dict['not_fishing_and_geared']=[]
  dumb_dict['not_fishing_and_not_geared']=[]
  for i in range(1,len(dict_log)+1):
    dumb_dict['people'].append(len(dict_log[i]))
    people = 0
    fishing =0
    not_fishing =0
    geared =0
    not_geared =0
    fishing_and_not_geared=0
    fishing_and_geared=0
    not_fishing_and_geared=0
    not_fishing_and_not_geared=0
    this_frame = dict_log[i]
    for j in range(len(this_frame)):
      if 'Fishing' in this_frame[j]:
        fishing+=1
        if 'Not Geared' in this_frame[j]:
          fishing_and_not_geared+=1
          not_geared+=1
        else:
          fishing_and_geared+=1
          geared+=1
      else:
        not_fishing+=1
        if 'Not Geared' in this_frame[j]:
          not_fishing_and_not_geared+=1
          not_geared+=1
        else:
          not_fishing_and_geared+=1
          geared+=1
    dumb_dict['fishing'].append(fishing)
    dumb_dict['not_fishing'].append(not_fishing)
    dumb_dict['geared'].append(geared)
    dumb_dict['not_geared'].append(not_geared)
    dumb_dict['fishing_and_not_geared'].append(fishing_and_not_geared)
    dumb_dict['fishing_and_geared'].append(fishing_and_geared)
    dumb_dict['not_fishing_and_geared'].append(not_fishing_and_geared)
    dumb_dict['not_fishing_and_not_geared'].append(not_fishing_and_not_geared)
  return dumb_dict


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


class FishingDetection:
    
    frame = 0
    max_frame = 0
    progress = 0
    jlog={}
    def __init__(self, human_detection_graph_path,
                 human_detection_graph_label_path,
                 num_classes,
                 fishing_classification_graph_path,
                 fishing_classification_graph_label_path,
                 geared_classification_graph_path):
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

        self.__geared_classification_graph_path = geared_classification_graph_path
        self.__geared_classification_graph = FishingDetection.load_inception(self.__geared_classification_graph_path)
        self.__sess =tf.Session(graph=self.__human_detection_graph)
        self.__fsess =tf.Session(graph=self.__fishing_classification_graph)
        self.__gsess =tf.Session(graph=self.__geared_classification_graph)
        self.__csess =tf.Session()


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
              output_dict = self.__sess.run(tensor_dict,
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
                   img):
        # print(img)
        img = self.make_square(img)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = np.asarray(img)
        img = (img/255)-0.5
        return self.__gear_model.predict(np.expand_dims(img, axis=0))

    def make_square(self,
                    im,
                    min_size=100,
                    fill_color=(0, 0, 0, 0)):
        im = Image.fromarray(im, 'RGB')
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        # print((int((size - x) / 2), int(size - y) / 2))
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    @staticmethod
    def load_trained_model_keras(weights_path):
        model = FishingDetection.nvidia_net()
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
        result = self.__csess.run(normalized)
        return result

    def pred_tensorflow_fishing(self,
                        image_np,
                        input_height=299,
                        input_width=299,
                        input_mean=0,
                        input_std=255):
        t = self.read_tensor_from_image_file(
            image_np,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)
        input_name = "import/" + "Placeholder"
        output_name = "import/" + "final_result"
        input_operation = self.__fishing_classification_graph.get_operation_by_name(input_name)
        output_operation = self.__fishing_classification_graph.get_operation_by_name(output_name)
        results = self.__fsess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)
        return results
    def pred_tensorflow_geared(self,
                        image_np,
                        input_height=299,
                        input_width=299,
                        input_mean=0,
                        input_std=255):
        t = self.read_tensor_from_image_file(
            image_np,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)
        input_name = "import/" + "Placeholder"
        output_name = "import/" + "final_result"
        input_operation = self.__geared_classification_graph.get_operation_by_name(input_name)
        output_operation = self.__geared_classification_graph.get_operation_by_name(output_name)
        results = self.__gsess.run(output_operation.outputs[0], {
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
        predictions1 = {}
        predictions1['bo'] = []
        predictions1['res'] = []
        predictions1['score'] = []
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > 0.3:
                predictions['bo'].append(boxes[i])
                predictions1['bo'].append(boxes[i])
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * im_height)
                xmin = int(xmin * im_width)
                ymax = int(ymax * im_height)
                xmax = int(xmax * im_width)
                roi = image_np[ymin:ymax, xmin:xmax]
                # c = pred(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                c = self.pred_tensorflow_fishing(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                d = self.pred_tensorflow_geared(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                if (c[0] > 0.5):
                    predictions['res'].append(0)
                    predictions['score'].append(c[0])
                else:
                    predictions['res'].append(1)
                    predictions['score'].append(c[1])
                if (d[0] > 0.5):
                    predictions1['res'].append(0)
                    predictions1['score'].append(d[0])
                else:
                    predictions1['res'].append(1)
                    predictions1['score'].append(d[1])    
        return predictions,predictions1

    def pipeline(self, img):
        FishingDetection.frame +=1
        FishingDetection.progress = (FishingDetection.frame/FishingDetection.max_frame)*100 
        image = Image.fromarray(img)
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = self.run_inference_for_single_image(image_np, self.__human_detection_graph)
        for i in range(len(output_dict['detection_classes'])):
            if output_dict['detection_classes'][i] != 1:
                output_dict['detection_scores'][i] = 0.0
        predi,predi1 = self.predict_fishing(image, output_dict['detection_boxes'], output_dict['detection_classes'],
                                     output_dict['detection_scores'])
        category_ind1 = {0: {'id': 0, 'name': 'Geared'}, 1: {'id': 1, 'name': 'Not Geared'}}
        category_ind = {1: {'id': 1, 'name': 'Not Fishing'}, 0: {'id': 0, 'name': 'Fishing'}}
        
        frame_log = make_frame_log(np.asarray(predi['bo'], dtype=np.float32),
            predi['res'],predi1['res'],
            category_ind,category_ind1,)
        
        FishingDetection.jlog[FishingDetection.frame]=frame_log
        
        visualize_boxes_and_labels_on_image_array(
            image_np,
            np.asarray(predi['bo'], dtype=np.float32),
            predi['res'],predi1['res'],
            predi['score'],predi1['score'],
            category_ind,category_ind1,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
        return image_np
    def close_all_sessions(self):
        self.__sess.close()
        self.__fsess.close()
        self.__gsess.close()
        self.__csess.close()
        
      
        


# In[ ]:


def main(INPUT_FILE,subclip_start,subclip_end):
    #Reset class variables to default values  
    FishingDetection.jlog={}
    FishingDetection.frame=0
    FishingDetection.max_frame=0
    FishingDetection.progress=0
    
    INPUT_DIRECTORY = 'input_video'
    OUTPUT_DIRECTORY = 'output_video'
#     INPUT_FILE = 'Catching_tuna_Maldivian_style.mp4'
    OUTPUT_FILE = 'result4.mp4'
    vid_output = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)
    vid_input = os.path.join(INPUT_DIRECTORY, INPUT_FILE)
#     subclip_start = '00:01:25.00'
#     subclip_end = '00:01:26.00'
    clip = VideoFileClip(vid_input).subclip(subclip_start, subclip_end)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    DIRECTORY_NAME = 'models'
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    GRAPH_NAME = 'frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = os.path.join(DIRECTORY_NAME, MODEL_NAME, GRAPH_NAME)

    NUM_CLASSES = 90
    fishing_detection_graph = os.path.join('fishing_model', 'output_graph.pb')
    geared_detection_graph = os.path.join('gear_detect_models', 'output_graph.pb')
    fishing_classification_graph_label_path = os.path.join('data', 'output_labels.txt')
    object_detect = FishingDetection(human_detection_graph_path=PATH_TO_FROZEN_GRAPH,
                                     human_detection_graph_label_path=PATH_TO_LABELS,
                                     num_classes=NUM_CLASSES,
                                     fishing_classification_graph_path=fishing_detection_graph,
                                     fishing_classification_graph_label_path=fishing_classification_graph_label_path,
                                     geared_classification_graph_path=geared_detection_graph)
    FishingDetection.max_frame = clip.duration * clip.fps 
    vid = clip.fl_image(object_detect.pipeline)
    vid.write_videofile(vid_output, audio=False)
    object_detect.close_all_sessions()
    with open('data.json', 'w') as fp:
      json.dump(FishingDetection.jlog, fp)
    new_log = make_dummer_log(FishingDetection.jlog)
    new_log['length'] = vid.duration
    with open('data_disp.json', 'w') as fp:
      json.dump(new_log, fp)
    


