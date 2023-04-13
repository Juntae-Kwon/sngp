from data import make_training_data, make_testing_data, make_ood_data
from sngp import snresnet, Laplace
from utils import mean_field_logits
from visualization import plot_predictions
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

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


# Based on the paper except for the number of epochs
num_inputs_features = 2
num_layers = 6
num_hidden = 128
dropout_rate = 0.01
num_outputs = 1
num_epochs = 70
batch_size = 128
lr = 1e-4
mean_field_factor = math.pi / 8.

dataset = data.TensorDataset(train_examples, train_labels)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Hidden representations
feature_extractor = snresnet(
    num_inputs_features=num_inputs_features,
    num_layers=num_layers,
    num_hidden=num_hidden,
    dropout_rate=dropout_rate,
    num_outputs=num_outputs
)

# According to the tensorflow implementation, the first layer is frozen
for param in feature_extractor.input_layer.parameters():
    param.requires_grad = False

model = Laplace(feature_extractor)

parameters = [
    {"params": model.parameters(), "lr": lr}
]

loss_function = nn.BCELoss()
optimizer = optim.Adam(parameters)

for epoch in range(num_epochs):
    model.train()
    losses = []
    accs = []
    print("\n- epoch: {}/{}".format(epoch+1, num_epochs))
    for samples in dataloader:
        x_train, y_train = samples    
        
        pred, covmat = model(x_train, return_gp_cov=True, update_cov=True)
        pred = mean_field_logits(pred, covmat, mean_field_factor=mean_field_factor)
        probabilities = torch.sigmoid(pred).squeeze()
        loss = loss_function(probabilities, y_train)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = (probabilities > 0.5).float()
        num_correct = (predictions == y_train).float()
        accs.append(torch.mean(num_correct))

    model.reset_cov()
    
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)

    print(f"average loss: {avg_loss:.4f}")
    print("average accuracy: ", avg_acc)


model.eval()
logits, cov = model(test_examples, return_gp_cov=True)
pred = mean_field_logits(logits, cov, mean_field_factor=mean_field_factor)
prob = 1-torch.sigmoid(pred)

plot_predictions(prob, model_name="SNGP")