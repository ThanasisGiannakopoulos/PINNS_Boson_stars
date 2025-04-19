# losses.py
import torch # type: ignore
from utils import gradients

def domain_loss(u, x, omega, m): 
    A, alpha, chi_minus, phi = map(lambda i:  u[:,[i]], range(4))
    #  s.t. all vars are positive
    chi = - chi_minus
    # take derivatives
    Ax = gradients(A, x)
    alphax = gradients(alpha, x)
    chix = gradients(chi, x)
    phix = gradients(phi, x)
    
    omega = omega#.to(device)
    
    # potential
    V = 0.5*torch.pow(m,2)*torch.pow(phi,2)
    # potential derivative wrt phi
    dVdphi = torch.pow(m,2)*phi
    # rho
    rho = 0.5*(torch.pow(chi,2)/A + torch.pow((omega/alpha),2)*torch.pow(phi,2)) + V
    # S_A
    SA = 0.5*(torch.pow(chi,2)/A + torch.pow((omega/alpha),2)*torch.pow(phi,2)) - V
    # eq_A is x*((1-x)^3)*A - rhs[A]
    eq_A = x*((1-x)**3)*Ax -((1-x)**2)*A*(1-A) - 8*torch.pi*(x**2)*(A**2)*rho
    # eq_alpha is x*((1-x)^3)*\p_x alpha - rhs[alpha]
    eq_alpha = x*((1-x)**3)*alphax - alpha*( 0.5*(A-1)*(1-x)**2 + 8*torch.pi*A*(x**2)*SA)
    # eq_chi  is x*((1-x)^2)*chi - rhs[chi]
    eq_chi = x*((1-x)**2)*chix + chi*( (1+A)*(1-x) - 8*torch.pi*x*A*V) - x*A*(dVdphi - torch.pow((omega/alpha),2)*phi)
    # eq_phi is ((1-x)^2)*\p_x phi - rhs[phi]
    eq_phi = ((1-x)**2)*phix - chi

    loss_dom = (torch.mean(torch.pow(eq_A,2)) + torch.mean(torch.pow(eq_alpha,2))
                + torch.mean(torch.pow(eq_chi,2)) + torch.mean(torch.pow(eq_phi,2)) 
               )
               
    return loss_dom

def x0_loss(u0, x0, phi0):    
    A, alpha, chi_minus, phi = map(lambda i:  u0[[i]], range(4))
    chi = - chi_minus
    # take derivatives
    alphax = gradients(alpha, x0)
    
    loss_x0 = torch.mean(torch.pow(A-1,2)) + torch.mean(torch.pow(alphax,2)) + torch.mean(torch.pow(phi-phi0,2)) + torch.mean(torch.pow(chi,2))
    return loss_x0

def x1_loss(u1):
    A, alpha, chi_minus, phi = map(lambda i:  u1[[i]], range(4))
    chi = - chi_minus

    loss_rmax = torch.mean(torch.pow(A-1,2)) + torch.mean(torch.pow(alpha-1,2)) + torch.mean(torch.pow(phi,2)) + torch.mean(torch.pow(chi,2))
    return loss_rmax

# impose that phi is a monotonically decreasing function
def phi_monotonic_decrease_dom(u,x):
    A, alpha, chi_minus, phi = map(lambda i:  u[:,[i]], range(4))
    chi = - chi_minus
    phix = gradients(phi, x)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(phix),2))
    return penalty

# impose that alpha is a monotonically increasing function
def alpha_monotonic_increase_dom(u,x):
    A, alpha, chi_minus, phi = map(lambda i:  u[:,[i]], range(4))
    chi = - chi_minus
    alphax = gradients(alpha, x)
    m = torch.nn.ReLU()
    penalty = torch.mean(torch.pow(m(-alphax),2))
    return penalty
