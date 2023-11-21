from PIL import Image
import torch

def test_single_image(image_path, model, device, transform):
    # Load image with PIL (ensure it's not in a "with" block so it's not closed after being converted to tensor)
    image = Image.open(image_path)
    
    # Apply the transformations and add an extra batch dimension
    def test_single_image(image_path, model, device, transform, test_dataset):
        # Load image with PIL (ensure it's not in a "with" block so it's not closed after being converted to tensor)
        image = Image.open(image_path)
        
        # Apply the transformations and add an extra batch dimension
        image = transform(image).unsqueeze(0).to(device)
        
        # Set model to evaluation mode and get the output
        model.eval()
        with torch.no_grad():
            output = model(image)
        
        # Convert output probabilities to predicted class
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        
        # Retrieve the class index
        index_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
        predicted_class_label = index_to_class[predicted_class]
        
        return predicted_class_label

    # Example usage:
    image_path = 'path_to_your_image.jpg'  # replace with your image path
    predicted_class_label = test_single_image(image_path, model, device, transform)
    print(f'Predicted class label: {predicted_class_label}')
    
    return predicted_class_label

# Example usage:
image_path = 'path_to_your_image.jpg'  # replace with your image path
predicted_class_label = test_single_image(image_path, model, device, transform)
print(f'Predicted class label: {predicted_class_label}')
