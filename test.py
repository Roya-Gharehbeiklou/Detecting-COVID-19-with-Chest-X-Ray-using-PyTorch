from torchvision import transforms, models
import torch
from PIL import Image

CLASS_NAMES = ['normal', 'viral', 'covid']

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(image_path):
    """Predict the class of a single image."""
    image = Image.open('COVID-19RadiographyDatabase/test/viral/Viral Pneumonia (57).png').convert('RGB')
    image = test_transform(image).unsqueeze(0)
    resnet18.eval()
    with torch.no_grad():
        output = resnet18(image)
        _, pred = torch.max(output, 1)
        return CLASS_NAMES[pred.item()]


resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=len(CLASS_NAMES))
resnet18.load_state_dict(torch.load('model.pth'))
resnet18.eval()

# for example
if __name__ == "__main__":
    result = predict('COVID-19RadiographyDatabase/test/viral/Viral Pneumonia (57).png')
    print(f'Predicted class: {result}')
