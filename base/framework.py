import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
import random


from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre

from tools.spatial_tools import make_mixed_EI_lattice
from tools.spatial_tools import distance_mask_2d_toroidal
from tools.build_W_in import build_tiled_gaussian_W_in
from tools.build_W_in import build_pixel_gaussian_W_in
from tools.other import sample_per_neuron_param
from tools.feature_encoding import encode_feature_map
from tools.feature_encoding import spikes_to_binned_counts

#metrics
from tools.metrics import calculate_CV
from tools.metrics import calculate_rate
from tools.metrics import calculate_rho_mean

#visualization
from visualization.visualizations import plot_raster
from visualization.visualizations import plot_rate_distribution
from visualization.visualizations import plot_spike_distribution
from visualization.visualizations_spatial import plot_EI_positions
from visualization.visualizations_spatial import plot_outgoing_connections
from visualization.visualizations_spatial import plot_spikecount_grid



class Framework:
    def __init__(self, n_neurons, time, dt, seed, bin_ms=50, log_normal=False, heterogeneity=False, mnist_input=True, self_tuning=True, spatial=True, convolution=True, input_channels=4, g=4, eta=1.0, sigma_input = 1, sigma_network = 1, epsilon = 0.3, intensity=430, stdp=False, nu_stdp=(1e-5, 1e-3), norm_stdp=None):
        self.seed = seed
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_default_device(self.device)
        print(f"Using device: {self.device}")
        
        # Simulation time parameters
        self.time = int(time)                                           
        self.dt = float(dt)  
        self.bin_ms = bin_ms
        # Methods
        self.heterogeneity = heterogeneity  
        self.intensity = intensity
        self.self_tuning = self_tuning
        self.spatial = spatial
        self.log_normal = log_normal
        self.mnist_input = mnist_input
        self.convolution = convolution
        self.stdp = stdp
        self.nu_stdp = nu_stdp
        self.norm_stdp = norm_stdp

        
        self.sigma_log = 0.8
        self.mu_log = math.log(1) - 0.5*self.sigma_log**2
        if self.spatial == False and self.convolution == True:
            raise ValueError("Cannot have Spatial is set to False and convolution to True.")
        if self.convolution:
            self.input_channels = 4
        else:
            self.input_channels = 1
        
        # Neuron Network parameters
        self.n_neurons = int(n_neurons) 
        self.frac_E = 0.8                               
        self.N_E = int(self.frac_E * self.n_neurons)                         
        self.N_I = self.n_neurons - self.N_E  

        # Brunel parameters
        self.g = g                                                 
        self.eta = eta   
        

        # Connectivity/synapse parameters
        self.epsilon_non_spatial = 0.1
        self.epsilon = epsilon
        self.sigma_input = sigma_input                                           
        self.sigma_network = sigma_network
        self.w_E = 1                                                 
        self.w_ext = 10  
        if mnist_input:
            self.w_input = 10.0
        else:
            self.w_input = 0.0

        self.mean_w_EE = self.w_E 
        self.mean_w_EI = self.w_E
        self.mean_w_IE = -self.g * self.w_E
        self.mean_w_II = -self.g * self.w_E
        weight_std = 0.2
        self.std_w_EE = weight_std * self.w_E
        self.std_w_EI = weight_std * self.w_E
        self.std_w_IE = weight_std * self.w_E
        self.std_w_II = weight_std * self.w_E

        # Sinlge neuron parameters
        self.theta_base = 20.0
        self.tau_m_base = 20.0
        if self.heterogeneity:
            self.theta_E = sample_per_neuron_param(self.N_E, base=self.theta_base, rel_std=0.20, min_val=5.0, max_val=40.0, device=self.device)
            self.tau_m_E = sample_per_neuron_param(self.N_E, base=self.tau_m_base, rel_std=0.25, min_val=5.0, max_val=60.0, device=self.device)
            self.theta_I = sample_per_neuron_param(self.N_I, base=self.theta_base, rel_std=0.20, min_val=5.0, max_val=40.0, device=self.device)
            self.tau_m_I = sample_per_neuron_param(self.N_I, base=self.tau_m_base, rel_std=0.25, min_val=5.0, max_val=60.0, device=self.device)
        else:
            self.theta_E = self.theta_base
            self.tau_m_E = self.tau_m_base
            self.theta_I = self.theta_base
            self.tau_m_I = self.tau_m_base
        self.tau_s = self.tau_m_base / 1000.0
        self.delay = 1.5 # synaptic delay in ms  
        self.delay_steps = int(self.delay / self.dt)                                        

        # Noise paramters
        self.v_th = self.theta_base / (self.tau_s * self.w_ext) 
        #self.rate_ext = self.eta * self.v_th
        self.rate_ext = self.eta*150.0#self.eta * self.theta_base / self.tau_s
        print(f"rate_ext: {self.rate_ext:.2f}")
        print(f"rate per step probability: {self.rate_ext * (self.dt/1000.0):.2f}")

    
    def build_network(self):
        print("Building network...")

        self.network = Network(dt=self.dt)

        self.neurons_E = LIFNodes(n=self.N_E, tau=self.tau_m_E, rest=0.0, reset=0.0, thresh=self.theta_E, refrac=1, traces=True, tc_trace = 20.0)
        self.neurons_I = LIFNodes(n=self.N_I, tau=self.tau_m_I, rest=0.0, reset=0.0, thresh=self.theta_I, refrac=1, traces=True, tc_trace = 20.0)
        self.network.add_layer(self.neurons_E, name="E")
        self.network.add_layer(self.neurons_I, name="I")

        # Noise Inputs
        self.noise_E = Input(n=self.N_E)
        self.noise_I = Input(n=self.N_I)
        self.network.add_layer(self.noise_E, name="noise_E")
        self.network.add_layer(self.noise_I, name="noise_I")

        if self.spatial:
            self.pos_all, self.pos_E, self.pos_I, self.rows, self.cols = make_mixed_EI_lattice(self.n_neurons, frac_E=self.frac_E, device=self.device)

            self.K = self.input_channels # number of feature maps (4 with gabor, 1 without)
            self.Hf = self.rows_f = 28
            self.Wf = self.cols_f = 28 
            self.D_in = self.K * self.Hf * self.Wf
            self.feat_in = Input(n=self.D_in, traces=True, tc_trace=20.0)
            self.network.add_layer(self.feat_in, name="F")
            self.W_in = build_tiled_gaussian_W_in(self.pos_E, self.N_E, self.rows, self.cols, self.K, self.Hf, self.Wf, self.sigma_input, margin = 0.5, w_input=self.w_input)
            if self.stdp:
                norm_val = self.norm_stdp if self.norm_stdp is not None else self.W_in.sum(dim=0).mean().item()
                self.connection_F_E = Connection(
                    source=self.feat_in, target=self.neurons_E, w=self.W_in,
                    update_rule=PostPre, nu=self.nu_stdp,
                    wmin=0.0, wmax=self.w_input * 2, norm=norm_val,
                )
                print(f"STDP norm (F→E): {norm_val:.2f}")
            else:
                self.connection_F_E = Connection(source=self.feat_in, target=self.neurons_E, w=self.W_in)
            self.network.add_connection(self.connection_F_E, source="F", target="E")

            self.mask_EE = distance_mask_2d_toroidal(self.pos_E, self.pos_E, self.epsilon_non_spatial, self.sigma_network, self.rows, self.cols, device=self.device)
            self.mask_EI = distance_mask_2d_toroidal(self.pos_E, self.pos_I, self.epsilon_non_spatial, self.sigma_network, self.rows, self.cols, device=self.device)
            self.mask_IE = distance_mask_2d_toroidal(self.pos_I, self.pos_E, self.epsilon_non_spatial, self.sigma_network, self.rows, self.cols, device=self.device)
            self.mask_II = distance_mask_2d_toroidal(self.pos_I, self.pos_I, self.epsilon_non_spatial, self.sigma_network, self.rows, self.cols, device=self.device)
        
        else:
            input_indices = torch.randint(0, 784, (self.N_E,)) # Which input neurons feeds excitatory neuron j
            self.W_in = torch.zeros(784, self.N_E)
            for j in range(self.N_E):
                i = input_indices[j]
                self.W_in[i, j] = float(self.w_input)
            self.mnist_in = Input(n=784, traces=True, tc_trace=20.0)
            self.network.add_layer(self.mnist_in, name="F")
            if self.stdp:
                norm_val = self.norm_stdp if self.norm_stdp is not None else self.W_in.sum(dim=0).mean().item()
                self.connection_F_E = Connection(
                    source=self.mnist_in, target=self.neurons_E, w=self.W_in,
                    update_rule=PostPre, nu=self.nu_stdp,
                    wmin=0.0, wmax=self.w_input * 2, norm=norm_val,
                )
                print(f"STDP norm (F→E): {norm_val:.2f}")
            else:
                self.connection_F_E = Connection(source=self.mnist_in, target=self.neurons_E, w=self.W_in)
            self.network.add_connection(self.connection_F_E, source="F", target="E")
            self.mask_EE = torch.bernoulli(torch.full((self.N_E, self.N_E), self.epsilon))
            self.mask_EI = torch.bernoulli(torch.full((self.N_E, self.N_I), self.epsilon))
            self.mask_IE = torch.bernoulli(torch.full((self.N_I, self.N_E), self.epsilon))
            self.mask_II = torch.bernoulli(torch.full((self.N_I, self.N_I), self.epsilon))

        if self.log_normal:
            log_samples = torch.normal(self.mu_log, self.sigma_log, size=(self.N_E, self.N_E))
            weights = torch.exp(log_samples)
            self.W_EE = self.mask_EE * weights
            self.W_EI = self.mask_EI * torch.normal(self.mean_w_EI, self.std_w_EI, size=(self.N_E, self.N_I))
            self.W_IE = self.mask_IE * torch.normal(self.mean_w_IE, self.std_w_IE, size=(self.N_I, self.N_E))
            self.W_II = self.mask_II * torch.normal(self.mean_w_II, self.std_w_II, size=(self.N_I, self.N_I))
        else:
            self.W_EE = self.mask_EE * torch.normal(self.mean_w_EE, self.std_w_EE, size=(self.N_E, self.N_E))
            self.W_EI = self.mask_EI * torch.normal(self.mean_w_EI, self.std_w_EI, size=(self.N_E, self.N_I))
            self.W_IE = self.mask_IE * torch.normal(self.mean_w_IE, self.std_w_IE, size=(self.N_I, self.N_E))
            self.W_II = self.mask_II * torch.normal(self.mean_w_II, self.std_w_II, size=(self.N_I, self.N_I))

        delay_EE = torch.full((self.N_E, self.N_E), self.delay_steps, dtype=torch.int)
        delay_EI = torch.full((self.N_E, self.N_I), self.delay_steps, dtype=torch.int)
        delay_IE = torch.full((self.N_I, self.N_E), self.delay_steps, dtype=torch.int)
        delay_II = torch.full((self.N_I, self.N_I), self.delay_steps, dtype=torch.int)

        connection_EE = Connection(source=self.neurons_E, target=self.neurons_E, w=self.W_EE.clone(), delay=delay_EE)
        connection_EI = Connection(source=self.neurons_E, target=self.neurons_I, w=self.W_EI, delay=delay_EI)
        connection_IE = Connection(source=self.neurons_I, target=self.neurons_E, w=self.W_IE, delay=delay_IE)
        connection_II = Connection(source=self.neurons_I, target=self.neurons_I, w=self.W_II, delay=delay_II)
        self.connection_IE = Connection(source=self.neurons_I, target=self.neurons_E, w=self.W_IE, delay=delay_IE)
        self.connection_II = Connection(source=self.neurons_I, target=self.neurons_I, w=self.W_II, delay=delay_II)
        connection_noise_E = Connection(source=self.noise_E, target=self.neurons_E, w=self.w_ext * torch.eye(self.N_E))
        connection_noise_I = Connection(source=self.noise_I, target=self.neurons_I, w=self.w_ext * torch.eye(self.N_I))

        self.network.add_connection(connection_EE, source="E", target="E")
        self.network.add_connection(connection_EI, source="E", target="I")
        self.network.add_connection(self.connection_IE, source="I", target="E")
        self.network.add_connection(self.connection_II, source="I", target="I")

        self.W_IE_base = self.connection_IE.w.clone()
        self.W_II_base = self.connection_II.w.clone()
        self.g_base = self.g
        self.network.add_connection(connection_noise_E, source="noise_E", target="E")
        self.network.add_connection(connection_noise_I, source="noise_I", target="I")

        # Monitors
        T = int(self.time / self.dt)
        self.mon_E = Monitor(self.neurons_E, state_vars=["s"], time=T)
        self.mon_I = Monitor(self.neurons_I, state_vars=["s"], time=T)
        self.network.add_monitor(self.mon_E, name="E_spikes")
        self.network.add_monitor(self.mon_I, name="I_spikes")

        self.network.learning = False
        print(f"Network built successfully (device: {self.device})")


    

    def run_one_sample(self, dataset, target):
        
        seed = 42
        random.seed(seed)
        idx = random.randrange(len(dataset))        
        while dataset[idx]["label"] != target:
            idx = random.randrange(len(dataset))

        sample = dataset[idx]
        label = sample["label"]
        feature_map = sample["feature_map"]
        feat_spikes = encode_feature_map(feature_map, self.time, self.dt, self.intensity)

        E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(feat_spikes)

        

        CV = calculate_CV(E_spikes, self.dt)
        print(f"CV: {CV:.2f}")
        rho_mean = calculate_rho_mean(E_spikes, self.dt)
        print(f"rho_mean: {rho_mean:.2f}")
        rate = calculate_rate(E_spike_counts, self.time)
        print(f"rate: {rate:.2f}")

        plot_raster(E_spikes, I_spikes, "Excitatory raster", "Inhibitory raster", g = self.g, eta = self.eta, CV = CV, rho_mean = rho_mean, rate = rate)
        plot_rate_distribution(self.time, E_spike_counts, I_spike_counts, "Excitatory rate distribution", "Inhibitory rate distribution")
        #plot_spike_distribution(E_spike_counts, I_spike_counts, "Excitatory and inhibitory spike distribution")
        plot_EI_positions(self.pos_E, self.pos_I)
        #plot_outgoing_connections(self.mask_EE, self.pos_E, 450, "Outgoing connections from excitatory neuron 450")
        plot_spikecount_grid(E_spike_counts, self.pos_E, "Excitatory spike count heatmap")
        

    
    def run_stimulation(self, dataset, examples, shuffle=True):
        

        pbar = tqdm(range(examples), desc=f"Stimulating network: (0 / {examples})")
        pairs = []
        CV_list, rho_mean_list, rate_list, g_list, eta_list = [], [], [], [], []

        for i, index in enumerate(pbar):
            if shuffle:
                index = random.choice(range(len(dataset)))
            sample = dataset[index]
            label = sample["label"]
            feature_map = sample["feature_map"]
            feat_spikes = encode_feature_map(feature_map, self.time, self.dt, self.intensity)

            E_spike_counts, I_spike_counts, E_spikes, I_spikes = self.run(feat_spikes)
            binned_E = spikes_to_binned_counts(E_spikes, bin_ms=self.bin_ms, dt=self.dt, time=self.time) # (N_bins, N_E)
            binned_E_flat = binned_E.flatten() # (N_bins * N_E,)
            features = binned_E_flat.float() # the features are the binned counts of the excitatory spikes in the network 
            pairs.append((features, label))

            CV_E = calculate_CV(E_spikes, self.dt)
            rho_mean_E = calculate_rho_mean(E_spikes, self.dt)
            rate_E = calculate_rate(E_spike_counts, self.time)

            CV_list.append(CV_E)
            rho_mean_list.append(rho_mean_E)
            rate_list.append(rate_E)
            g_list.append(self.g)
            eta_list.append(self.eta)

            pbar.set_description_str(f"Stimulating network: ({i+1} / {examples})")

            if self.self_tuning:
                self._self_tune(CV_E, rho_mean_E, rate_E)
            

        return pairs, CV_list, rho_mean_list, rate_list, g_list, eta_list


    def run(self, feat_spikes):
        feat_spikes = feat_spikes.to(self.device)
        encoder = PoissonEncoder(time=1, dt=self.dt)

        for t in range(self.time):
            rates_XE = torch.ones(self.N_E, device=self.device) * self.rate_ext
            rates_XI = torch.ones(self.N_I, device=self.device) * self.rate_ext
            spikes_XE = encoder(rates_XE)
            spikes_XI = encoder(rates_XI)
            feat_t = feat_spikes[t:t+1] # (1, 1, D_in)
            self.network.run(inputs={"noise_E": spikes_XE.unsqueeze(0), "noise_I": spikes_XI.unsqueeze(0), "F": feat_t}, time=1)

        E_spikes = self.mon_E.get("s") # shape (T, 1, N_E)
        I_spikes = self.mon_I.get("s") # shape (T, 1, N_I)
        E_spike_counts = E_spikes.squeeze(1).sum(0) # shape (N_E,)
        I_spike_counts = I_spikes.squeeze(1).sum(0) # shape (N_I,)

        # Reset state variables
        self.network.reset_state_variables()
        self.mon_E.reset_state_variables()
        self.mon_I.reset_state_variables()

        return E_spike_counts, I_spike_counts, E_spikes, I_spikes



    def enable_stdp(self):
        self.network.learning = True

    def disable_stdp(self):
        self.network.learning = False

    def run_stdp_training(self, dataset, n_samples, shuffle=True):
        self.W_in_before = self.connection_F_E.w.detach().clone()
        self.enable_stdp()

        pbar = tqdm(range(n_samples), desc="STDP Training")
        for i in pbar:
            idx = random.choice(range(len(dataset))) if shuffle else i
            sample = dataset[idx]
            feature_map = sample["feature_map"]
            feat_spikes = encode_feature_map(feature_map, self.time, self.dt, self.intensity)
            self.run(feat_spikes)

        self.disable_stdp()

        self.W_in_after = self.connection_F_E.w.detach().clone()
        delta = (self.W_in_after - self.W_in_before).abs().mean().item()
        print(f"STDP training complete. Mean |ΔW|: {delta:.6f}")

    def plot_input_weights(self, title="Input Weights", n_show=16):
        w = self.connection_F_E.w.detach().clone().cpu().numpy()

        indices = np.linspace(0, self.N_E - 1, n_show, dtype=int)
        cols = int(np.ceil(np.sqrt(n_show)))
        rows = int(np.ceil(n_show / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            rf = w[:, idx]
            if self.spatial:
                rf = rf.reshape(self.K, self.Hf, self.Wf).sum(axis=0)
            else:
                rf = rf.reshape(28, 28)
            axes[i].imshow(rf, cmap='hot', interpolation='nearest')
            axes[i].set_title(f'N{idx}', fontsize=8)
            axes[i].axis('off')

        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        safe_title = title.replace(' ', '_').lower()
        plt.savefig(f"results/{safe_title}.png", dpi=150, bbox_inches='tight')
        plt.show()

    def plot_weight_change(self, n_show=16):
        if not hasattr(self, 'W_in_before') or not hasattr(self, 'W_in_after'):
            print("No STDP training has been run yet.")
            return

        delta = (self.W_in_after - self.W_in_before).cpu().numpy()
        indices = np.linspace(0, self.N_E - 1, n_show, dtype=int)
        cols = int(np.ceil(np.sqrt(n_show)))
        rows = int(np.ceil(n_show / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        vmax = np.abs(delta).max()
        for i, idx in enumerate(indices):
            rf = delta[:, idx]
            if self.spatial:
                rf = rf.reshape(self.K, self.Hf, self.Wf).sum(axis=0)
            else:
                rf = rf.reshape(28, 28)
            axes[i].imshow(rf, cmap='RdBu_r', interpolation='nearest',
                           vmin=-vmax, vmax=vmax)
            axes[i].set_title(f'N{idx}', fontsize=8)
            axes[i].axis('off')

        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        fig.suptitle("STDP Weight Change (ΔW)", fontsize=14)
        plt.tight_layout()
        plt.savefig("results/stdp_weight_change.png", dpi=150, bbox_inches='tight')
        plt.show()


    def set_g(self, new_g):
        scale = float(new_g / self.g_base)
        with torch.no_grad():
            self.connection_IE.w.copy_(self.W_IE_base * scale)
            self.connection_II.w.copy_(self.W_II_base * scale)
        self.g = new_g

    def set_eta(self, new_eta):
        self.eta = new_eta
        self.rate_ext = self.v_th * self.eta

    def _self_tune(self, CV, rho, rate):
        CV_target = 0.8
        rho_target = 0.02
        rate_min_alive = 1.0       # below this, network is dead
        rate_target_low = 10.0
        rate_target_high = 40.0
        rate_max_runaway = 100.0
        
        CV_tol = 0.2
        rho_tol = 0.02
        
        k_g = 0.5
        k_eta = 0.1
        
        g_min, g_max = 2.0, 20.0
        eta_min, eta_max = 0.5, 5.0
        
        # --- Priority 0: is the network alive? ---
        if rate < rate_min_alive:
            # Silent. Ignore CV and rho — they're meaningless here.
            # Push drive up, don't touch g.
            new_eta = self.eta + k_eta * max((rate_min_alive - rate) / rate_min_alive, 0.1)
            self.set_eta(float(np.clip(new_eta, eta_min, eta_max)))
            return
        
        # --- Priority 1: runaway / saturation ---
        if rate > rate_max_runaway:
            # Too hot. Pull drive down first.
            err = (rate - rate_max_runaway) / rate_max_runaway
            self.set_eta(float(np.clip(self.eta - k_eta * err, eta_min, eta_max)))
            # If it's also synchronous, add inhibition
            if rho > rho_target:
                self.set_g(float(np.clip(self.g + k_g * (rho - rho_target), g_min, g_max)))
            return
        
        # --- Now CV and rho are meaningful ---
        
        # Priority 2: regime control via g
        rho_excess = max(rho - rho_target, 0.0)
        CV_deficit = max(CV_target - CV_tol - CV, 0.0)
        regime_error = max(rho_excess, CV_deficit)

        # Is rate badly wrong? If so, fix it even if regime isn't perfect.
        rate_badly_low = rate < rate_target_low * 0.5   # e.g. < 5 Hz
        rate_badly_high = rate > rate_target_high * 1.5  # e.g. > 60 Hz
        rate_urgent = rate_badly_low or rate_badly_high

        g_saturated = abs(self.g - g_max) < 1e-6 or abs(self.g - g_min) < 1e-6
        if regime_error > rho_tol and not rate_urgent and not g_saturated:
            # Regime is off and rate isn't critical — tune g
            self.set_g(float(np.clip(self.g + k_g * regime_error, g_min, g_max)))
            return

        # Rate control (runs if in regime, OR if rate is urgent)
        if rate > rate_target_high:
            self.set_eta(np.clip(self.eta - k_eta * (rate - rate_target_high) / rate_target_high, eta_min, eta_max))
        elif rate < rate_target_low:
            self.set_eta(np.clip(self.eta + k_eta * (rate_target_low - rate) / rate_target_low, eta_min, eta_max))
        


    def get_configuration_info(self):
        print("======== Quick Summary of the parameters ========")

        # Methods
        print("\n----- Methods -----")
        print(f"Heterogeneity: {self.heterogeneity}")
        print(f"Intensity: {self.intensity}")
        print(f"Self tuning: {self.self_tuning}")
        print(f"MNIST input: {self.mnist_input}")

        # Simulation parameters
        print("----- Simulation Parameters -----")
        print(f"Simulation time per sample [ms]: {self.time}")
        print(f"Time step [ms]: {self.dt}")

        # Network structure
        print("\n----- Network Structure -----")
        print(f"Total number of neurons: {self.n_neurons}")
        print(f"Number of excitatory neurons: {self.N_E}")
        print(f"Number of inhibitory neurons: {self.N_I}")

        # Connectivity/synapse parameters
        print("\n----- Connectivity/Synapse Parameters -----")
        print(f"Connection probability: {self.epsilon}")
        print(f"Relative inhibitory strength (g): {self.g}")
        print(f"eta (external rate multiplier): {self.eta}")
        print(f"(Excitatory ->) synapse weight: {self.w_E}")
        print(f"(Input ->) synapse weight: {self.w_input}")
        print(f"(Noise ->) synapse weight: {self.w_ext}")

       

        # Neuron parameters
        print("\n----- Neuron Parameters -----")
        print(f"Membrane threshold potential (theta) [mV]: {self.theta}")
        print(f"Membrane time constant (tau_m) [ms]: {self.tau_m}")

        # Rates
        print("\n----- Rates -----")
        print(f"Threshold rate (v_th): {self.v_th}")
        print(f"External rate (v_ext): {self.rate_ext}")

        # Miscellaneous
        print("\n----- Seed -----")
        print(f"Seed: {self.seed}")

        print("--------------------------------")
        
        print("--------------------------------")