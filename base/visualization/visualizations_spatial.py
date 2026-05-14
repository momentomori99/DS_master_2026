import torch
import matplotlib.pyplot as plt

def plot_EI_positions(pos_E, pos_I):
        """
        Plot excitatory and inhibitory neuron positions
        on the same 2D toroidal grid.
        """
        pE = pos_E.detach().cpu().numpy()
        pI = pos_I.detach().cpu().numpy()

        plt.figure(figsize=(5, 5))

        # Excitatory neurons
        plt.scatter(
            pE[:, 1], pE[:, 0],
            s=8, c="tab:blue", label="Excitatory", alpha=0.8
        )

        # Inhibitory neurons
        plt.scatter(
            pI[:, 1], pI[:, 0],
            s=12, c="tab:red", label="Inhibitory", alpha=0.9
        )

        plt.gca().invert_yaxis()
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.title("E / I neuron positions on toroidal lattice")
        plt.legend(markerscale=1.5)
        plt.tight_layout()
        plt.show(block=True)
        plt.close()


def plot_outgoing_connections(mask: torch.Tensor, pos_post: torch.Tensor, pre_idx: int, title=None):
        """
        mask: (N_pre, N_post) 0/1
        pos_post: (N_post, 2)
        """
        m = mask[pre_idx].detach().cpu()  # (N_post,)
        p = pos_post.detach().cpu()
        connected = (m > 0)

        plt.figure(figsize=(5,5))
        plt.scatter(p[:,1], p[:,0], s=8, alpha=0.2, label="all post")
        plt.scatter(p[connected,1], p[connected,0], s=15, alpha=0.9, label="connected")
        plt.gca().invert_yaxis()
        plt.title(title or f"Outgoing connections from pre neuron {pre_idx}")
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)
        plt.close()


def plot_spikecount_grid(counts: torch.Tensor, pos: torch.Tensor, title="Spike count heatmap"):
        counts = counts.detach().cpu().float()
        pos = pos.detach().cpu().float()

        # infer grid size from positions
        rows = int(pos[:, 0].max().item()) + 1
        cols = int(pos[:, 1].max().item()) + 1

        grid = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(pos.shape[0]):
            r = int(pos[i, 0].item())
            c = int(pos[i, 1].item())
            grid[r, c] = counts[i]

        plt.figure(figsize=(6, 5))
        plt.imshow(grid.numpy(), aspect="equal")
        plt.title(title)
        plt.xlabel("x (col)")
        plt.ylabel("y (row)")
        plt.colorbar(label="spikes")
        plt.tight_layout()
        plt.show(block=True)
        plt.close()