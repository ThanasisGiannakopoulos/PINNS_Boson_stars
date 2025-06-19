# losses.py
import torch # type: ignore
from utils import gradients

"""
We implement the Dirichlet boundary conditions at r=0 and r=rmax in a hard enforcement, meaning
we change the variable that the NN approximates, s.t. the BC at r=0 is exactly satisfied.

1. 
A = 1 + sin(pi*r/rmax)*NN_A; (1=r^0 to make sure derivatives can be taken)
-> A(0) = A(rmax) = 1, since sin(0) = sin(pi) = 0.
A > 0, since sin(pi*r/rmax) > 0 and NN_A > 0, due to softmax enforced in NN output.

2. 
phi = phi0*sin[(pi/2)(r/rmax + 1)] + sin(pi *r/rmax)*NN_phi
-> phi(0) = phi0 and phi(rmax) = 0
phi(r) > 0  since NN_phi > 0 from softmax and the sin's used are also > 0 in their chosen domain

3.
chi = sin[pi(1+r/rmax)]*NN_chi
-> chi(0) = chi(rmax) = 0
and chi < 0, since sin[pi(1+r/rmax)] < 0 and NN_chi > 0

4.
alpha = 1 - sin[(pi/2)*(1 + r/rmax)]*NN_alpha
-> alpha[rmax] = 1
and alpha(r) <= 1 for r \in [0,rmax], since NN_alpha > 0

Also we need soft constr. for \p_r alpha |r=0 = 0. with this redef we need to impose
d/dr * NN_alpha |r=0 = 0 (since sin and cos will give 0 and 1 at r=0).
"""

def domain_loss(u, r, omega, m, phi_0, rmax):
    # load nn approximates
    nn_A, nn_alpha, nn_chi_minus, nn_phi = map(lambda i:  u[:, [i]], range(4))
    # define pi
    pi = torch.pi
    # use the nn approximates to construct the unknowns
    A = r**0 + torch.sin(pi*r/rmax)*nn_A
    alpha = r**0 - torch.sin(0.5*pi*(1.0 + r/rmax))*nn_alpha
    chi = torch.sin(pi*(1.0 + r/rmax))*nn_chi_minus
    phi = phi_0*torch.sin(0.5*pi*(1.0 + r/rmax)) + torch.sin(pi*r/rmax)*nn_phi
    # take derivatives w.r.t. r 
    # explicitly for unknowns; derivs w.r.t. of graphs only for nn_approxs
    # A
    Ar = gradients(A, r)
    # alpha
    alphar = gradients(alpha, r)
    # chi
    chir = gradients(chi, r)
    # phi
    phir = gradients(phi, r)
    
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
    nn_alpha = u0[1]
    nn_alphar = gradients(nn_alpha, r0)

    loss_r0 = torch.mean(torch.pow(nn_alphar, 2))
    return loss_r0

def phi_monotonic_decrease_dom(u, r, phi_0, rmax):
    # define pi
    pi = torch.pi
    # load nn approximates
    nn_phi = u[:, [3]]
    # use the nn approximates to construct the unknowns
    phi = phi_0*torch.sin(0.5*pi*(1.0 + r/rmax)) + torch.sin(pi*r/rmax)*nn_phi
    # take derivative w.r.t r
    phir = gradients(phi, r)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(phir), 2))
    return penalty

def alpha_monotonic_increase_dom(u, r, rmax):
    # define pi
    pi = torch.pi
    nn_alpha = u[:, [1]]
    alpha = r**0 - torch.sin(0.5*pi*(1.0 + r/rmax))*nn_alpha
    alphar = gradients(alpha, r)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(-alphar), 2))
    return penalty
