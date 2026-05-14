import math 
import torch

def make_mixed_EI_lattice(N_total: int, frac_E: float = 0.8, device="cpu"):
        cols = int(math.ceil(math.sqrt(N_total)))
        rows = int(math.ceil(N_total / cols))

        rr, cc = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing="ij"
        )
        pos_all = torch.stack([rr.flatten(), cc.flatten()], dim=1).float()  # (rows*cols, 2)
        pos_all = pos_all[:N_total]  # keep exactly N_total sites

        # random assignment of sites to E/I
        perm = torch.randperm(N_total, device=device)
        N_E = int(round(frac_E * N_total))
        idx_E = perm[:N_E]
        idx_I = perm[N_E:]

        pos_E = pos_all[idx_E]
        pos_I = pos_all[idx_I]

        return pos_all, pos_E, pos_I, rows, cols

def distance_mask_2d_toroidal(pos_pre, pos_post, epsilon, sigma, rows, cols, device="cpu"):
        # pos_* are (N, 2) with (row, col)

        drow = torch.abs(pos_pre[:, None, 0] - pos_post[None, :, 0])
        dcol = torch.abs(pos_pre[:, None, 1] - pos_post[None, :, 1])

        # wrap-around distances
        drow = torch.minimum(drow, rows - drow)
        dcol = torch.minimum(dcol, cols - dcol)

        d2 = drow**2 + dcol**2

        P = torch.exp(-d2 / (2.0 * sigma * sigma))

        # For recurrent connections, remove self connections if want
        #if pos_pre.shape[0] == pos_post.shape[0]:
        #    P.fill_diagonal_(0.0)

        meanP = P.mean()
        if meanP.item() > 1e-12:
            P = P * (epsilon / meanP)

        P = P.clamp(0.0, 1.0)
        mask = torch.bernoulli(P).to(device)
        return mask
    