import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from gflownet.envs.pruning import Model, ModelArchitecture
from tqdm import tqdm
import os

def get_pretrained_model(model_type='lenet', num_classes=10):
    """Get a pretrained model and convert it to our ModelArchitecture format."""
    # Automatically extract architecture
    layer_params = []
    model = torchvision.models.resnet18(pretrained=True)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer_params.append({
                "type": "conv2d",
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size[0],
                "stride": layer.stride[0],
                "padding": layer.padding[0],
            })
        elif isinstance(layer, nn.BatchNorm2d):
            layer_params.append({
                "type": "batch_norm",
                "num_features": layer.num_features,
            })
        elif isinstance(layer, nn.ReLU):
            layer_params.append({
                "type": "relu"
            })
        elif isinstance(layer, nn.Dropout):
            layer_params.append({
                "type": "dropout",
                "p": layer.p,
            })
        elif isinstance(layer, nn.Linear):
            layer_params.append({
                "type": "linear",
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            })
    
    print(layer_params)
    architecture = ModelArchitecture(layer_params=layer_params)
    torch.save(model.state_dict(), 'models/lenet5_pretrained.pt')
    return model, architecture

def train_and_evaluate_lenet(epochs=10, batch_size=128, save_path='models/lenet5_cifar10_pretrained.pt'):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transform for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        './data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Architecture for CIFAR-10
    model, architecture = get_pretrained_model(model_type='lenet', num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        for data, target in tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')
        
        # Evaluate the model on training data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(trainloader, desc="Evaluating Training"):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        train_accuracy = 100 * correct / total
        print(f'Training Accuracy after epoch {epoch + 1}: {train_accuracy:.2f}%')

        # Evaluate the model on test data after each epoch
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(testloader, desc="Evaluating Test"):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_accuracy = 100 * correct / total
        test_loss /= len(testloader)
        print(f'Test Accuracy after epoch {epoch + 1}: {test_accuracy:.2f}%')
        print(f'Test Loss after epoch {epoch + 1}: {test_loss:.3f}')

    print('Finished Training')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return model, test_accuracy

if __name__ == '__main__':
    get_pretrained_model()
