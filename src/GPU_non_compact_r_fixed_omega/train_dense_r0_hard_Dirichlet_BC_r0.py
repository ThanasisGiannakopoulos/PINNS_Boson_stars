# train.py
import os
import torch # type: ignore
import numpy as np # type: ignore

from model import FCN
from losses_hard_Dirichlet_BC_r0 import domain_loss, r0_loss, rmax_loss, phi_monotonic_decrease_dom, alpha_monotonic_increase_dom
from utils import random_domain_points
from reset_lr import reset_lr_scheduler

# for the model
neurons = 64
h_layers = 4
RMAX = 1000
# Number of random domain points (n)
n = 1000  # Adjust this as necessary
# for grid to be more dense near r=0
sigma = 0.1

# Configuration and hyperparameters
out_dir = f"../../models/GPU_non_compact_r_fixed_omega/hard_Dirichlet_BC_r0_neurons{neurons}_h_layers{h_layers}_rmax{RMAX}_n{n}_sigma{sigma}/"
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
epochs = 1000 #100000  # Total number of epochs
save_model_every = 10000 
current_lr = initial_lr  # Set current learning rate to the initial learning rate

# print message every # steps
message_every = 1

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

    # the loss at r0
    r0 = torch.zeros(1, requires_grad=True).to(device)
    u0 = model(r0)
    loss_r0 = r0_loss(u0, r0)
    # the bulk loss and monotonicity penalties
    #r = RMAX * random_domain_points(n).to(device)
    pts = random_domain_points(n).to(device) # uniform points in (0,1)
    r = RMAX * torch.sinh(pts/sigma)/torch.sinh(torch.ones(1).to(device)/sigma)
    u = model(r)
    loss_dom = domain_loss(u, r, omega, m, phi0, RMAX)
    phi_mono_decrease = phi_monotonic_decrease_dom(u, r, phi0, RMAX)
    alpha_mono_increase = alpha_monotonic_increase_dom(u, r)
    # loss at rmax
    rmax = RMAX * torch.ones(1, requires_grad=True).to(device)
    umax = model(rmax)
    loss_rmax = rmax_loss(umax, phi0)

    # Total loss
    loss = (w0 * loss_r0 + w1 * loss_dom + w2 * loss_rmax + w3 * phi_mono_decrease + w4 * alpha_mono_increase)
    
    # save individual losses
    losses[0].append(loss_r0.item())
    losses[1].append(loss_dom.item())
    losses[2].append(loss_rmax.item())
    losses[3].append(phi_mono_decrease.item())
    losses[4].append(alpha_mono_increase.item())
    #print(losses[1])
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

    # # Every `reset_interval` epochs, reset the learning rate and apply linear decay to the reseted lr
    # if (epoch + 1) % reset_interval == 0:
    #     # Reset the learning rate
    #     current_lr = reset_lr_scheduler(optimizer, epoch, reset_interval, initial_lr, final_reset_lr, num_resets)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = current_lr

    #     print(f"Learning rate reset to {current_lr:.6e} at epoch {epoch + 1}")

# save model
#torch.save(model.state_dict(), out_dir + "model.pth")
# save losses and learning rate lists
np.savez(out_dir + "losses.npz", loss=losses)
np.savez(out_dir + "total_loss.npz", loss=loss_list)
np.savez(out_dir + "learning_rate.npz", lr=lr_list)

