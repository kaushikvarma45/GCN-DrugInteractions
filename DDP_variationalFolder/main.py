import Adjacency_vae
from Adjacency_vae import MoleculeDataset, reconstruction_accuracy, GVAE
from Adjacency_vae import gvae_loss
import classifier
from classifier import NumbersDataset,NeuralNet,TestDataset
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score


data = pd.read_csv('raw/totalData.csv')
encoder_set = MoleculeDataset("","totalData.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GVAE(feature_size=encoder_set[0].x.shape[1],edge_size=encoder_set[0].edge_attr.shape[1])


data_size = len(encoder_set)
NUM_GRAPHS_PER_BATCH = 64
frac = 0.8
train_loader = DataLoader(encoder_set[:int(data_size * frac)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(encoder_set[int(data_size * frac):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


model = model.to(device)

loss_fn = gvae_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
kl_beta = 0.005




def run_one_epoch(data_loader, type, epoch, kl_beta):
    # Store per batch loss and accuracy 
    all_losses = []
    all_accs = []

    # Iterate over data loader
    for i, batch in enumerate(tqdm(data_loader)):
        # Some of the data points have invalid adjacency matrices 
        try:
            # Use GPU
            batch.to(device)  
            # Reset gradients
            optimizer.zero_grad() 
            # Call model
            triu_logits, mu, logvar = model(batch.x.float(), 
                                            batch.edge_attr.float(),
                                            batch.edge_index, 
                                            batch.batch) 
            # Calculate loss and backpropagate
            loss, kl_div = loss_fn(triu_logits, batch.edge_index, mu, logvar, batch.batch, kl_beta)
            if type == "Train":
                loss.backward()  
                optimizer.step()  
            # Calculate metrics
            acc = reconstruction_accuracy(triu_logits, batch.edge_index, batch.batch, batch.x.float())
            

            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            all_accs.append(acc)
            
            

        except IndexError as error:
            # For a few graphs the edge information is not correct
            # Simply skip the batch containing those
            print("Error: ", error)
    
    print(f"{type} epoch {epoch} loss: ", np.array(all_losses).mean())
    print(f"{type} epoch {epoch} accuracy: ", np.array(all_accs).mean())
   

  

for epoch in range(6): 
  model.train()
  run_one_epoch(train_loader, type="Train", epoch=epoch, kl_beta=kl_beta)
  if epoch % 5 == 0:
      print("test epoch...")
      model.eval()
      run_one_epoch(test_loader, type="Test", epoch=epoch, kl_beta=kl_beta)


NUM_GRAPHS_PER_BATCH = 1
loader = DataLoader(encoder_set, 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)

dictval = {}
for batch in loader : 
  batch.to(device)
  decode_pred, mu, logvar = model(batch.x.float(), 
                            batch.edge_attr.float(),
                            batch.edge_index, 
                            batch.batch) 
  
  
  dictval[batch.id[0]] = [mu,logvar]
  
print("done!!!")



dftrain = pd.read_csv("sup_train_val.csv")
dftest = pd.read_csv("sup_test.csv")
n_classes = len(dftrain['label'].unique())

dataset = NumbersDataset("sup_train_val.csv",dictval)
data = dataset



model_ff = NeuralNet(data[0][0].shape[0],128)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ff.parameters(), lr=0.001)  
model_ff = model_ff.to(device)
loss_fn = loss_fn.to(device)
# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 128
loader = DataLoader(data,batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)



def train(data):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch[0] = batch[0].to(device) 
      batch[1] = batch[1].to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      
      pred = model_ff(batch[0]) 
      
      # Calculating the loss and gradients
      
      
      loss = (loss_fn(pred, batch[1]))      
      loss.backward()  
      # Update using the gradients
      optimizer.step()   
    return loss

print("Starting training...")
losses = []
for epoch in range(251):
    loss = train(data)
    losses.append(loss)
    if epoch % 50 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")


datatest = TestDataset("sup_test.csv")

test_loader = DataLoader(datatest,batch_size=len(datatest), shuffle=True)
dffull = pd.DataFrame(columns = ['y_real','y_pred'])
with torch.no_grad():
    for test_batch in test_loader : 
      test_batch[0] = test_batch[0].to(device)
      test_batch[1] = test_batch[1].to(device)
      pred = model_ff(test_batch[0]) 
      pred_frac0 = pred[:,0]
      pred_frac1 = pred[:,1]
      if n_classes == 3 : 
        pred_frac2 = pred[:,2]
      pred = pred.argmax(dim=1)  
      dftt = pd.DataFrame()
      dftt["y_real"] = test_batch[1].tolist()
      dftt["y_pred"] = pred.tolist()
      dftt["y_predfrac0"] = pred_frac0.tolist()
      dftt["y_predfrac1"] = pred_frac1.tolist()
      if n_classes == 3 : 
        dftt["y_predfrac2"] = pred_frac2.tolist()
      dffull = dffull.append(dftt)
      print(f1_score(dftt['y_real'], dftt['y_pred'],average='macro'))

print("Done classifier!!!")