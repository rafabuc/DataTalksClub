import onnxruntime as ort

import torchvision.transforms as transforms


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])


session = ort.InferenceSession(
    "hair_classifier_empty.onnx", providers=["CPUExecutionProvider"]
)
#input_name = session.get_inputs()[0].name
#output_name = session.get_outputs()[0].name

classes = [
    "curly",
    "straight"
]


def predict(url):
    #X = preprocessor.from_url(url)
    img = download_image(url)
    img = prepare_image(img, (200, 200))

    img_tensor = train_transforms(img)

    print(f"Shape del tensor: {img_tensor.shape}")  # torch.Size([3, 200, 200])
    #print(f"Tipo del tensor: {img_tensor}")  # torch.float32
    img_batch = img_tensor.unsqueeze(0)
    print(f"Shape del batch: {img_batch.shape}")  # torch.Size([1, 3, 200, 200])

    input_name = session.get_inputs()[0].name
    print(f"Nombre de entrada: {input_name}")

    # Convertir el tensor de PyTorch a numpy
    img_numpy = img_batch.numpy()  # o img_batch.detach().numpy() si tiene gradientes

    # Ejecutar inferencia
    outputs = session.run(None, {input_name: img_numpy})

    #result = session.run([output_name], {input_name: img_batch})
    float_predictions = outputs[0][0].tolist()
    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
    