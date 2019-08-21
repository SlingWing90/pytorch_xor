import torch
import torch.optim as optim

#### Custom Network Module
class ANN(torch.nn.Module):
	def __init__(self):
		super(ANN, self).__init__()
		
		self.fc1 = torch.nn.Linear(2, 3)
		self.fc2 = torch.nn.Linear(3, 1)
		
	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		
		return x

# Load Model from local "my_model.py"
load_model = False
		
# DATASET
X = torch.tensor(([0, 0],[0, 1],[1, 0], [1, 1]), dtype=torch.float)
y = torch.tensor(([0], [1], [1], [0]), dtype=torch.float)
	
#### LOAD FROM TRAINING
if load_model == True:
	nn = torch.load("./my_model.py")
	nn.eval()
#### TRAINING
else: 
	# START
	nn = ANN()

	optimizer = optim.SGD(nn.parameters(), lr=0.1) # Stochastic Gradient Descent
	criterion = torch.nn.MSELoss() # Criterion to Measure the mean squared error 

	epochs = 6000 # Count of Epochs

	for e in range(epochs):
		for x in range(0, 4):
			out = nn(X[x]) # 

			optimizer.zero_grad() # Set Gradient to Zero
			
			loss = criterion(out, y[x]) # Calculate Loss
			
			loss.backward() # Backpropagation
			
			optimizer.step() # Update Weight
			
			params = list(nn.parameters())
		else:
			print("Epoch: "+str(e)) # Print out actual Epoch
####

# Print Out Result
print("Result:")		
for _x, _y in zip(X, y):
    prediction = nn(_x)
    print('Input:\t', list(map(int, _x)))
    print('Pred:\t', int(prediction >= 0.5))
    print('Ouput:\t', int(_y))
    print('######')

# Save Model
torch.save(nn, "./my_model.py")
