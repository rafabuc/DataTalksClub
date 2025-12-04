import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    
    # Convertir a numpy y normalizar (reemplaza torchvision.transforms)
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # Especificar float32
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)    # Especificar float32

    img_array = (img_array - mean) / std
    
    # Cambiar de HWC a CHW (Height, Width, Channels -> Channels, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Agregar dimensión del batch
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_batch

session = ort.InferenceSession(
    "hair_classifier_empty.onnx", 
    providers=["CPUExecutionProvider"]
)

classes = ["curly", "straight"]

def predict(url):
    img = download_image(url)
    img_batch = prepare_image(img)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_batch})
    
    float_predictions = outputs[0][0].tolist()
    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result