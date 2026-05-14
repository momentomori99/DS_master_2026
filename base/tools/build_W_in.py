import math
import torch

def build_tiled_gaussian_W_in(pos_E, N_E, rows, cols, K, Hf, Wf, sigma_in, margin, w_input):
        D_in = K * Hf * Wf
        W_in = torch.zeros(D_in, N_E, dtype=torch.float32)

        # tiling layout (rows x cols grid split into tile_rows x tile_cols)
        tile_rows = int(math.floor(math.sqrt(K)))
        tile_cols = int(math.ceil(K / tile_rows))

        tile_h = rows / tile_rows
        tile_w = cols / tile_cols

        for k in range(K):
            tr_idx = k // tile_cols
            tc_idx = k % tile_cols

            r0 = tr_idx * tile_h
            r1 = (tr_idx + 1) * tile_h
            c0 = tc_idx * tile_w
            c1 = (tc_idx + 1) * tile_w

            # keep targets away from borders
            r0m, r1m = r0 + margin, r1 - margin
            c0m, c1m = c0 + margin, c1 - margin

            for u in range(Hf):
                ur = u / (Hf - 1) if Hf > 1 else 0.5
                for v in range(Wf):
                    vr = v / (Wf - 1) if Wf > 1 else 0.5

                    i = k * (Hf * Wf) + u * Wf + v

                    tr = r0m + ur * (r1m - r0m)
                    tc = c0m + vr * (c1m - c0m)

                    # Toroidal distance keeps neighborhood behavior consistent at borders.
                    dr = torch.abs(pos_E[:, 0] - tr)
                    dc = torch.abs(pos_E[:, 1] - tc)
                    dr = torch.minimum(dr, rows - dr)
                    dc = torch.minimum(dc, cols - dc)
                    d2 = dr ** 2 + dc ** 2
                    W_in[i, :] = torch.exp(-d2 / (2 * sigma_in**2))

        # Normalize each input row so every pixel has one strongest E target
        # and weaker neighbors with Gaussian falloff.
        row_max = W_in.max(dim=1, keepdim=True).values
        W_in /= (row_max + 1e-12)
        W_in *= w_input
        return W_in


def build_pixel_gaussian_W_in(pos_E, N_E, rows, cols, H=28, W=28, sigma_in=1.0, margin=0.5, w_input=1.0):
        """
        Non-convolution input mapping: 1 channel of raw pixels (H x W) -> E neurons.
        Each pixel (u, v) maps to the proportionate spatial location in the E lattice:
            u in [0, H-1] -> row in [margin, rows-1-margin]
            v in [0, W-1] -> col in [margin, cols-1-margin]
        and connects to E neurons with a non-toroidal Gaussian footprint.
        This preserves image topology, so with low/no noise the E activity map
        resembles the original digit pattern.
        """
        D_in = H * W
        W_in = torch.zeros(D_in, N_E, dtype=torch.float32)

        r0m, r1m = margin, (rows - 1) - margin
        c0m, c1m = margin, (cols - 1) - margin

        for u in range(H):
            ur = u / (H - 1) if H > 1 else 0.5
            tr = r0m + ur * (r1m - r0m)
            for v in range(W):
                vr = v / (W - 1) if W > 1 else 0.5
                tc = c0m + vr * (c1m - c0m)
                i = u * W + v

                # Non-toroidal euclidean distance for faithful image geometry.
                dr = pos_E[:, 0] - tr
                dc = pos_E[:, 1] - tc
                d2 = dr ** 2 + dc ** 2
                W_in[i, :] = torch.exp(-d2 / (2 * sigma_in ** 2))

        # Normalize per pixel so each pixel has a clear strongest E target.
        row_max = W_in.max(dim=1, keepdim=True).values
        W_in = W_in / (row_max + 1e-12)
        W_in = W_in * w_input
        return W_in