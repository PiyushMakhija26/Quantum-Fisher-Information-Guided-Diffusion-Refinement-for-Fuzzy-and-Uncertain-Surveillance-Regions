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
        
        self.I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        self.H_matrix = self._build_hamiltonian()
        self.H_sq_matrix = torch.matmul(self.H_matrix, self.H_matrix)

    def _build_hamiltonian(self):
        N = self.k * self.k
        H = torch.zeros((2**N, 2**N), dtype=torch.complex64, device=self.device)
        
        # Transverse Field (-h * X)
        for i in range(N):
            op = self.X if i == 0 else self.I
            for j in range(1, N):
                op = torch.kron(op, self.X if j == i else self.I)
            H -= self.h * op
            
        # Interaction Term (-J * ZZ)
        def add_interaction(i, j):
            op = self.Z if (i == 0 or j == 0) else self.I
            for idx in range(1, N):
                op = torch.kron(op, self.Z if (idx == i or idx == j) else self.I)
            return op

        for row in range(self.k):
            for col in range(self.k):
                i = row * self.k + col
                if col < self.k - 1: # Right neighbor
                    H -= self.J * add_interaction(i, i + 1)
                if row < self.k - 1: # Bottom neighbor
                    H -= self.J * add_interaction(i, i + self.k)
        return H

    def _encode_tensor_product_state(self, patch):
        """
        Maps N pixels to N local qubits. 
        |phi_a> = sqrt(1-r_a)|0> + sqrt(r_a)|1>
        |Psi> = Tensor Product of all |phi_a>
        """
        N = self.k * self.k
        state = torch.tensor([1.0 + 0j], device=self.device)
        for i in range(N):
            r_a = patch[i]
            qubit = torch.stack([torch.sqrt(1.0 - r_a), torch.sqrt(r_a)]).to(dtype=torch.complex64)
            state = torch.kron(state, qubit)
        return state

    def generate_mask(self, image_tensor):
        B, C, H_img, W_img = image_tensor.shape
        unfold = torch.nn.Unfold(kernel_size=self.k, stride=self.k)
        patches = unfold(image_tensor).transpose(1, 2).squeeze(0)
        
        qfi_raw = []
        for patch in patches:
            psi = self._encode_tensor_product_state(patch)
            
            E = torch.vdot(psi, torch.mv(self.H_matrix, psi)).real
            E_sq = torch.vdot(psi, torch.mv(self.H_sq_matrix, psi)).real
            
            variance = E_sq - (E ** 2)
            qfi_raw.append((4 * variance).item())
            
        qfi_tensor = torch.tensor(qfi_raw, device=self.device).view(H_img//self.k, W_img//self.k)
        qfi_tensor = F.interpolate(qfi_tensor.unsqueeze(0).unsqueeze(0), size=(H_img, W_img), mode='nearest').squeeze()
        
        mu_f, sigma_f = qfi_tensor.mean(), qfi_tensor.std()
        normalized_mask = 1.0 / (1.0 + torch.exp((qfi_tensor - mu_f) / (self.gamma * sigma_f)))
        
        return normalized_mask
