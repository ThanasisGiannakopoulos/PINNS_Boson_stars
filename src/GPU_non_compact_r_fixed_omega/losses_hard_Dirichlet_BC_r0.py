# losses.py
import torch # type: ignore
from utils import gradients

"""
We implement the Dirichlet boundary conditions at r=0 in a hard enforcement, meaning
we change the variable that the NN approximates, s.t. the BC at r=0 is exactly satisfied.

Example: A(r=0)=1. 
Instead of asking the NN to approximate A (denoted here as nn_A) we define A in the following way:
A = 1 + r*nn_A. This A is the one that goes in the domain_loss function and nn_A is the output the nn returns.

Similarly for phi and chi:
phi = phi_0 + r*nn_phi; phi(r=0) = phi_0
chi = - r*nn_chi; chi(r=0)=0 (the nn approximates -r*nn_chi <0 s.t. we can also enforce the monotonicity condition later)
"""
def domain_loss(u, r, omega, m, phi_0, Rmax): 
    # load nn approximates
    nn_A, alpha, nn_chi_minus, nn_phi = map(lambda i:  u[:, [i]], range(4))
    # use the nn approximates to construct the unknowns
    A = 1 + r*nn_A/Rmax
    chi = - r*nn_chi_minus/Rmax
    phi = phi_0 + r*nn_phi/Rmax
    # take derivatives w.r.t. r 
    # explicitly for unknowns; derivs w.r.t. of graphs only for nn_approxs
    # A
    nn_A_r = gradients(nn_A, r)
    Ar = nn_A/Rmax + r*nn_A_r/Rmax 
    # alpha
    alphar = gradients(alpha, r)
    # chi
    nn_chi_minus_r = gradients(nn_chi_minus, r)
    chir = - nn_chi_minus/Rmax - r*nn_chi_minus_r/Rmax
    # phi
    nn_phi_r = gradients(nn_phi, r)
    phir = nn_phi/Rmax + r*nn_phi_r/Rmax
    
    omega = omega#.to(device)

    V = 0.5 * torch.pow(m, 2) * torch.pow(phi, 2)
    dVdphi = torch.pow(m, 2) * phi
    rho = 0.5 * (torch.pow(chi, 2) / A + torch.pow((omega / alpha), 2) * torch.pow(phi, 2)) + V
    SA = 0.5 * (torch.pow(chi, 2) / A + torch.pow((omega / alpha), 2) * torch.pow(phi, 2)) - V
    eq_A = r * Ar - A * ((1 - A) + 8 * torch.pi * (r ** 2) * A * rho)
    eq_alpha = r * alphar - alpha * (0.5 * (A - 1) + 8 * torch.pi * A * (r ** 2) * SA)
    eq_chi = r * chir + (chi) * (1 + A - 8 * torch.pi * r * A * V) - r * A * (dVdphi - torch.pow((omega / alpha), 2) * phi)
    eq_phi = phir - chi

    # print("eq_A=", eq_A)
    
    # print("eq A mean = ", torch.mean(torch.pow(eq_A, 2)))
    # print("eq alpha mean = ", torch.mean(torch.pow(eq_alpha, 2)))
    # print("eq chi mean = ", torch.mean(torch.pow(eq_chi, 2)))
    # print("eq phi mean = ", torch.mean(torch.pow(eq_phi, 2)))
    
    loss_dom = (torch.mean(torch.pow(eq_A, 2)) + torch.mean(torch.pow(eq_alpha, 2))
                + torch.mean(torch.pow(eq_chi, 2)) + torch.mean(torch.pow(eq_phi, 2)))
    return loss_dom

def r0_loss(u0, r0):    
    alpha = u0[1]
    alphar = gradients(alpha, r0)

    loss_r0 = torch.mean(torch.pow(alphar, 2))
    return loss_r0

def rmax_loss(umax, phi_0):
    # load nn approximates
    nn_A, alpha, nn_chi_minus, nn_phi = map(lambda i:  umax[i], range(4))
    # use the nn approximates to construct the unknowns
    chi = - nn_chi_minus
    A = 1 + nn_A
    phi = phi_0 + nn_phi
    
    loss_rmax = torch.mean(torch.pow(A-1, 2)) + torch.mean(torch.pow(alpha-1, 2)) + torch.mean(torch.pow(phi, 2)) + torch.mean(torch.pow(chi, 2))
    return loss_rmax

def phi_monotonic_decrease_dom(u, r, phi_0, Rmax):
    # load nn approximates
    nn_phi = u[:, [3]]
    # use the nn approximates to construct the unknowns
    phi = phi_0 + r*nn_phi/Rmax
    # take derivative w.r.t r
    nn_phi_r = gradients(nn_phi, r)
    phir = nn_phi + r*nn_phi_r
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(phir), 2))
    return penalty

def alpha_monotonic_increase_dom(u, r):
    alpha = u[:, [1]]
    alphar = gradients(alpha, r)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(-alphar), 2))
    return penalty
