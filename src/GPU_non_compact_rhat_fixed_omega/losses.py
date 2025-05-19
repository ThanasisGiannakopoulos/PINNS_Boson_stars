# losses.py
import torch # type: ignore
from utils import gradients

def domain_loss(u, rhat, omega, m, Rmax):
    A, alpha, chi_minus, phi = map(lambda i:  u[:, [i]], range(4))
    chi = - chi_minus
    A_rhat = gradients(A, rhat)
    alpha_rhat = gradients(alpha, rhat)
    chi_rhat = gradients(chi, rhat)
    phi_rhat = gradients(phi, rhat)

    omega = omega#.to(device)

    V = 0.5 * torch.pow(m, 2) * torch.pow(phi, 2)
    dVdphi = torch.pow(m, 2) * phi
    rho = 0.5 * (torch.pow(chi, 2) / A + torch.pow((omega / alpha), 2) * torch.pow(phi, 2)) + V
    SA = 0.5 * (torch.pow(chi, 2) / A + torch.pow((omega / alpha), 2) * torch.pow(phi, 2)) - V
    eq_A = rhat * A_rhat - A * ((1 - A) + 8 * torch.pi * ( (Rmax * rhat) ** 2) * A * rho)
    eq_alpha = rhat * alpha_rhat - alpha * (0.5 * (A - 1) + 8 * torch.pi * A * ( (Rmax * rhat) ** 2) * SA)
    eq_chi = rhat * chi_rhat + chi * (1 + A - 8 * torch.pi * Rmax * rhat * A * V) - Rmax* rhat * A * (dVdphi - torch.pow((omega / alpha), 2) * phi)
    eq_phi = phi_rhat - Rmax * chi

    loss_dom = (torch.mean(torch.pow(eq_A, 2)) + torch.mean(torch.pow(eq_alpha, 2))
                + torch.mean(torch.pow(eq_chi, 2)) + torch.mean(torch.pow(eq_phi, 2)))
    return loss_dom

def r0_loss(u0, rhat0, phi0):
    A, alpha, chi_minus, phi = map(lambda i:  u0[[i]], range(4))
    chi = - chi_minus
    alpha_rhat = gradients(alpha, rhat0)

    loss_r0 = torch.mean(torch.pow(A-1, 2)) + torch.mean(torch.pow(alpha_rhat, 2)) + torch.mean(torch.pow(phi - phi0, 2)) + torch.mean(torch.pow(chi, 2))
    return loss_r0

def rmax_loss(umax):
    A, alpha, chi_minus, phi = map(lambda i:  umax[[i]], range(4))
    chi = - chi_minus

    loss_rmax = torch.mean(torch.pow(A-1, 2)) + torch.mean(torch.pow(alpha-1, 2)) + torch.mean(torch.pow(phi, 2)) + torch.mean(torch.pow(chi, 2))
    return loss_rmax

def phi_monotonic_decrease_dom(u, rhat):
    A, alpha, chi_minus, phi = map(lambda i:  u[:, [i]], range(4))
    chi = - chi_minus
    phi_rhat = gradients(phi, rhat)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(phi_rhat), 2))
    return penalty

def alpha_monotonic_increase_dom(u, rhat):
    A, alpha, chi_minus, phi = map(lambda i:  u[:, [i]], range(4))
    chi = - chi_minus
    alpha_rhat = gradients(alpha, rhat)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(-alpha_rhat), 2))
    return penalty
