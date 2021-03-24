import os
import tflite_runtime.interpreter as tflite
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from PIL import Image
from skimage import transform
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

column_names = ['beagle','chihuahua','doberman','french_bulldog','golden_retriever',
 'malamute',
 'pug',
 'saint_bernard',
 'scottish_deerhound',
 'tibetan_mastiff']

# create view here

def base64_file(data, name=None):
	_format, _img_str = data['img'].split(';base64,')
	print(_format)
	_name, ext = _format.split('/')
	if not name:
		name = _name.split(":")[-1]
	_path = default_storage.save('{}.{}'.format(name, ext), ContentFile(base64.b64decode(_img_str), name='{}.{}'.format(name, ext)))
	return _path

def load(filename):
	filename = os.path.join(BASE_DIR,'media\\'+filename)
	np_image = Image.open(filename)
	np_image = np.array(np_image).astype('float32')/255
	np_image = transform.resize(np_image, (224, 224, 3))
	np_image = np.expand_dims(np_image, axis=0)
	return np_image


@api_view(["POST"])
def breed(request):
	try:

		imgData = request.data
		imgData = load(base64_file(imgData))
		model_path = os.path.join(BASE_DIR,'tmpcizky496.tflite')
		interpreter = tflite.Interpreter(model_path)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		interpreter.set_tensor(input_details[0]['index'], imgData)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		id = int(np.argmax(output_data, axis = 1))
		return JsonResponse('breed : {} score : {}'.format(column_names[id],max(output_data[0])), safe=False)

	except ValueError as e:
		return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

