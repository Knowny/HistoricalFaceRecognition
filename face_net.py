# filename: face_net.py
# pretrained model: InceptionResnetV1
# This code uses the InceptionResnetV1 model from the facenet-pytorch library to compute face embeddings
# project: KNN Face Recognition
# version: 1.0


from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity


model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0) # add batch dimension
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding[0].numpy()

image_path1 = "path"
image_path2 = "path"

embedding1 = get_embedding(image_path1)
embedding2 = get_embedding(image_path2)

# cosine similarity

similarity = cosine_similarity(embedding1.reshape(1,-1), embedding2.reshape(1,-1)) # scikit takes 2D arrays
distance = 1 - similarity

print(f"Similarity: {similarity}")
print(f"Distance: {distance}")