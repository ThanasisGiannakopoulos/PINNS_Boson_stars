# losses.py
import torch # type: ignore
from utils import gradients

def domain_loss(u, r, omega, m): 
    A, alpha, chi_minus, phi = map(lambda i:  u[:, [i]], range(4))
    chi = - chi_minus
    Ar = gradients(A, r)
    alphar = gradients(alpha, r)
    chir = gradients(chi, r)
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

    loss_dom = (torch.mean(torch.pow(eq_A, 2)) + torch.mean(torch.pow(eq_alpha, 2))
                + torch.mean(torch.pow(eq_chi, 2)) + torch.mean(torch.pow(eq_phi, 2)))
    return loss_dom

def r0_loss(u0, r0, phi0):    
    A, alpha, chi_minus, phi = map(lambda i:  u0[[i]], range(4))
    chi = - chi_minus
    alphar = gradients(alpha, r0)

    loss_r0 = torch.mean(torch.pow(A-1, 2)) + torch.mean(torch.pow(alphar, 2)) + torch.mean(torch.pow(phi - phi0, 2)) + torch.mean(torch.pow(chi, 2))
    return loss_r0

def rmax_loss(umax):
    A, alpha, chi_minus, phi = map(lambda i:  umax[[i]], range(4))
    chi = - chi_minus

    loss_rmax = torch.mean(torch.pow(A-1, 2)) + torch.mean(torch.pow(alpha-1, 2)) + torch.mean(torch.pow(phi, 2)) + torch.mean(torch.pow(chi, 2))
    return loss_rmax

def phi_monotonic_decrease_dom(u, r):
    A, alpha, chi_minus, phi = map(lambda i:  u[:, [i]], range(4))
    chi = - chi_minus
    phir = gradients(phi, r)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(phir), 2))
    return penalty

def alpha_monotonic_increase_dom(u, r):
    A, alpha, chi_minus, phi = map(lambda i:  u[:, [i]], range(4))
    chi = - chi_minus
    alphar = gradients(alpha, r)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(-alphar), 2))
    return penalty
