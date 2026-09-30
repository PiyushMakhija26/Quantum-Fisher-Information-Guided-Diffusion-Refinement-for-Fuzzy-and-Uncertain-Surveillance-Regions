import torch
import torch.nn.functional as F
import numpy as np

class QFIMaskGenerator:
    def __init__(self, k=4, J=1.0, h=0.5, gamma=1.0, device='cuda'):
        """
        k: Patch size (4x4 yields 16 local qubits, Hilbert space 2^16).
        J, h: Ising parameters treated as empirical hyperparameters (J/h = 2.0).
        """
        self.k = k
        self.J = J
        self.h = h
        self.gamma = gamma
        self.device = device
        self.N = k * k
        
        # Precompute lattice adjacency and bond topology for 2D TFIM
        bonds = []
        for row in range(k):
            for col in range(k):
                i = row * k + col
                if col < k - 1:  # Right neighbor
                    bonds.append((i, i + 1))
                if row < k - 1:  # Bottom neighbor
                    bonds.append((i, i + k))
        self.bonds = bonds
        self.bonds_i = torch.tensor([b[0] for b in bonds], dtype=torch.long, device=device)
        self.bonds_j = torch.tensor([b[1] for b in bonds], dtype=torch.long, device=device)
        
        # Precompute bond pairs sharing a site for 2-bond cross covariance
        shared_list, other1_list, other2_list = [], [], []
        for b1_idx in range(len(bonds)):
            for b2_idx in range(b1_idx + 1, len(bonds)):
                b1 = set(bonds[b1_idx])
                b2 = set(bonds[b2_idx])
                common = b1.intersection(b2)
                if len(common) == 1:
                    shared = list(common)[0]
                    other1 = list(b1 - common)[0]
                    other2 = list(b2 - common)[0]
                    shared_list.append(shared)
                    other1_list.append(other1)
                    other2_list.append(other2)
                    
        self.shared = torch.tensor(shared_list, dtype=torch.long, device=device)
        self.other1 = torch.tensor(other1_list, dtype=torch.long, device=device)
        self.other2 = torch.tensor(other2_list, dtype=torch.long, device=device)

    def generate_mask(self, image_tensor):
        """
        Generates deterministic QFI reliability mask via analytical tensor contractions
        over separable product states |Psi> = \bigotimes |phi_a>, avoiding OOM from 2^16 x 2^16 dense matrices.
        """
        image_tensor = image_tensor.to(self.device)
        B, C, H_img, W_img = image_tensor.shape
        unfold = torch.nn.Unfold(kernel_size=self.k, stride=self.k)
        
        # patches shape: (num_patches, N)
        patches = unfold(image_tensor).squeeze(0).transpose(0, 1).clamp(0.0, 1.0)
        
        # Pauli single-site expectation values for |phi_a> = sqrt(1-r_a)|0> + sqrt(r_a)|1>
        # <X_a> = 2 * sqrt(r_a * (1 - r_a))
        # <Z_a> = 1 - 2 * r_a
        x = 2.0 * torch.sqrt(patches * (1.0 - patches) + 1e-12)
        z = 1.0 - 2.0 * patches
        
        # 1. Transverse field variance: Var(-h \sum X_i) = h^2 \sum (1 - <X_i>^2) = h^2 \sum <Z_i>^2
        var_A = (self.h**2 * (z**2)).sum(dim=-1)
        
        # 2. Nearest-neighbor interaction variance: Var(-J \sum Z_i Z_j) = J^2 \sum (1 - <Z_i>^2 <Z_j>^2)
        var_B = (self.J**2 * (1.0 - (z[:, self.bonds_i]**2) * (z[:, self.bonds_j]**2))).sum(dim=-1)
        
        # 3. Field-Interaction cross covariance: 2 * Cov(-h X_i, -J Z_j Z_k) = -2 * h * J * <Z_j> <Z_k> (<X_j> + <X_k>)
        cov_AB = (-self.h * self.J * z[:, self.bonds_i] * z[:, self.bonds_j] * (x[:, self.bonds_i] + x[:, self.bonds_j])).sum(dim=-1)
        
        # 4. Interaction-Interaction cross covariance for bonds sharing a common vertex
        cov_BB = (self.J**2 * z[:, self.other1] * z[:, self.other2] * (x[:, self.shared]**2)).sum(dim=-1)
        
        # Total Hamiltonian variance: Var(H) = Var_A + Var_B + 2*Cov_AB + 2*Cov_BB
        variance = var_A + var_B + 2.0 * cov_AB + 2.0 * cov_BB
        qfi_raw = 4.0 * variance
        
        # Spatially reshape and interpolate back to image resolution
        qfi_tensor = qfi_raw.view(1, 1, H_img // self.k, W_img // self.k)
        qfi_tensor = F.interpolate(qfi_tensor, size=(H_img, W_img), mode='nearest').squeeze()
        
        # Temperature-scaled logistic normalization
        mu_f, sigma_f = qfi_tensor.mean(), qfi_tensor.std()
        normalized_mask = 1.0 / (1.0 + torch.exp((qfi_tensor - mu_f) / (self.gamma * (sigma_f + 1e-8))))
        
        return normalized_mask
