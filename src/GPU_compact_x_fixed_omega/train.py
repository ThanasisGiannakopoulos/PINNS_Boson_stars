# train.py
import os
import torch # type: ignore
import numpy as np # type: ignore

from model import FCN
from losses import domain_loss, x0_loss, x1_loss, phi_monotonic_decrease_dom, alpha_monotonic_increase_dom
from utils import random_domain_points

# for the model
neurons = 64
h_layers = 4
# Number of random domain points (n)
n = 1000  # Adjust this as necessary
# for grid to be more dense near r=0
#sigma = 0.0

# Configuration and hyperparameters
#out_dir = f"../../models/GPU_compact_x_fixed_omega/neurons{neurons}_h_layers{h_layers}_n{n}_sigma{sigma}/"
out_dir = f"../../models/GPU_compact_x_fixed_omega/neurons{neurons}_h_layers{h_layers}_n{n}/"
#f"../../models/GPU_compact_x_fixed_omega/neurons{neurons}_h_layers{h_layers}_n{n}/"
os.makedirs(out_dir, exist_ok=True)

print('torch version:',torch.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

#device = torch.device("cpu")
#print('device:', device)

model = FCN(1, 4, neurons, h_layers).to(device)

omega = 0.895042 * torch.ones(1).to(device)
phi0  = 0.05  * torch.ones(1).to(device)
m = torch.ones(1).to(device)

# Initial learning rate
initial_lr = 1e-3  # Starting learning rate
#final_reset_lr = 1e-5  # finale learning rate in resets
#reset_interval = 100000  # Interval after which learning rate will reset
epochs = 100000  # Total number of epochs
save_model_every = 10000 
current_lr = initial_lr  # Set current learning rate to the initial learning rate

# print message every # steps
message_every = 100

# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

# Learning rate scheduler for exponential decay after reset
DECAY_RATE = 0.95
DECAY_STEPS = 20000
gamma = DECAY_RATE ** (1 / DECAY_STEPS)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Track the number of resets
#num_resets = epochs // reset_interval

# Define weights for each loss term
w0 = 1000.0  # Weight for r0_loss
w1 = 1.0  # Weight for domain_loss
w2 = 1000.0  # Weight for rmax_loss
w3 = 1000.0  # Weight for phi_monotonic_decrease
w4 = 1000.0  # Weight for alpha_monotonic_increase

# Loss and training loop
losses = [[], [], [], [], []]
loss_list = []
lr_list = []

for epoch in range(epochs):
    optimizer.zero_grad()

    # the loss at x0
    x0 = torch.zeros(1, requires_grad=True).to(device)
    u0 = model(x0)
    loss_x0 = x0_loss(u0, x0, phi0)
    # the bulk loss and monotonicity penalties
    pts = random_domain_points(n).to(device) # uniform points in (0,1)
    #x = torch.sinh(pts/sigma)/torch.sinh(torch.ones(1).to(device)/sigma)
    x = pts
    u = model(x)
    loss_dom = domain_loss(u, x, omega, m)
    phi_mono_decrease = phi_monotonic_decrease_dom(u, x)
    alpha_mono_increase = alpha_monotonic_increase_dom(u, x)
    # loss at x1=1
    x1 = torch.ones(1, requires_grad=True).to(device)
    ux1 = model(x1)
    loss_x1 = x1_loss(ux1)

    # Total loss
    loss = (w0 * loss_x0 + w1 * loss_dom + w2 * loss_x1 + w3 * phi_mono_decrease + w4 * alpha_mono_increase)
    
    # save individual losses
    losses[0].append(loss_x0.item())
    losses[1].append(loss_dom.item())
    losses[2].append(loss_x1.item())
    losses[3].append(phi_mono_decrease.item())
    losses[4].append(alpha_mono_increase.item())
    # save total loss
    loss_list.append(loss.item())
    # save current learning rate
    current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
    lr_list.append(current_lr)

    # print message
    if (epoch + 1) % message_every == 0:
        print(f"epoch = {epoch+1} | loss = {loss.item():.6e} | learning rate = {current_lr:.6e} |")#,  end='\r')

    if (epoch + 1) % save_model_every == 0:
        torch.save(model.state_dict(), out_dir + f"model_epoch{epoch+1}.pth")

    # Adjust learning rate based on exponential decay (after each reset)
    loss.backward()
    optimizer.step()
    scheduler.step()

# save model
# save losses and learning rate lists
np.savez(out_dir + "losses.npz", loss=losses)
np.savez(out_dir + "total_loss.npz", loss=loss_list)
np.savez(out_dir + "learning_rate.npz", lr=lr_list)

