from data import make_training_data, make_testing_data, make_ood_data
from deepresnet_deterministic import DeepResnet
from visualization import plot_uncertainty_surface
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt


DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_N_GRID = 100

train_examples, train_labels = make_training_data(sample_size=500)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)

print(train_examples.shape) # 1000 by 2 
print(train_labels.shape) # 1000
print(test_examples.shape) # 10000 by 2
print(ood_examples.shape) # 500 by 2

resnet_model = DeepResnet(num_inputs_features=2, num_layers=6, num_hidden=128, dropout_rate=0.01, num_outputs=1)

# According to the tensorflow implementation, the first layer is frozen
for param in resnet_model.input_layer.parameters():
    param.requires_grad = False

loss_function = nn.BCELoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=1e-4)
num_epochs = 100
batch_size = 128

dataset = data.TensorDataset(train_examples, train_labels)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    losses = []
    accs = []
    print("\n- epoch: {}/{}".format(epoch+1, num_epochs))
    for samples in dataloader:
        x_train, y_train = samples    
        
        probabilities = torch.sigmoid(resnet_model(x_train)).squeeze()
        loss = loss_function(probabilities, y_train)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = (probabilities > 0.5).float()
        num_correct = (predictions == y_train).float()
        accs.append(torch.mean(num_correct))
    
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)

    print(f"average loss: {avg_loss:.4f}")
    print("average accuracy: ", avg_acc)


# Now visualize the predictions of the deterministic model. First plot the class probability
resnet_logits = resnet_model(test_examples)
resnet_probs = 1-torch.sigmoid(resnet_logits) # Take the probability for class 0

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(resnet_probs, ax=ax)

plt.colorbar(pcm, ax=ax)
plt.title("Class Probability, Deterministic Model")

plt.show()