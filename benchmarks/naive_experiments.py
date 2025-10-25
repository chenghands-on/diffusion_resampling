import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from itertools import cycle
import torchsummary
import os
import gc

# Create results directory structure
os.makedirs('results/IS', exist_ok=True)
print("Importance Sampling results will be saved to: results/bounded/")


parser = argparse.ArgumentParser(description="Train your diffusion model.")
parser.add_argument("--n", type=int, default=1000, help="Number of target samples 10, 100, 1000")
parser.add_argument("--eta_cycle", type=float, default=0.0, help="Batch size")
parser.add_argument("--eta_consistency", type=float, default=0.0, help="Number of epochs")
parser.add_argument("--show_baseline", action="store_true", help="Show the baseline plots")
parser.add_argument("--show_density_ratio", action="store_true", help="Show the density ratio plots")
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda or mps)")
parser.add_argument("--resampling_strategy", type=str, default='every_step', 
                   choices=['every_step', 'sparse'], 
                   help='Resampling strategy: every_step (1,2,3,...,25) or sparse (2,4,8,16)')
args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# # Define constants
# d = 2
# sigma2 = 0.1
# mu_S = torch.tensor([0.5, 0.5], dtype=torch.float32)
# mu_T = torch.tensor([0.5, 0.5], dtype=torch.float32)
# # assert torch.dot(mu_S, mu_T) == 0
# m = 10000 # Number of source samples
# n = args.n # Number of target samples 10, 100, 1000


diffusion_steps = 25
device = args.device
# Training parameters
beta_0 = 0.1
beta_1 = 20.0
learning_rate = 1e-4
batch_size = 4096
target_batch_size = 256
# # epochs
c_more_epochs = 1
vanilla_diffusion_epochs = 50*c_more_epochs
finetune_diffusion_epochs = 50*c_more_epochs
guidance_epochs = 20*c_more_epochs
source_epochs = 100*c_more_epochs

# Sampling
guidance_scale = [1, 1]
show_baseline = args.show_baseline
# Eta1 is the Cycle Loss and Eta2 is the Consistency Loss
# eta1and2 = [eta1, eta2] = [0, 0] for no regularization
eta1and2 = [args.eta_cycle, args.eta_consistency]
use_regularization = True if sum(eta1and2) > 0 else False
show_density_ratio = args.show_density_ratio

# # -----------------------------------------------------------------------
# # 1. Generate samples from source and target domains
# # -----------------------------------------------------------------------
# def generate_samples(mu, sigma2, num_samples):
#     """Generate samples from a single Gaussian distribution"""
#     dist = torch.distributions.MultivariateNormal(mu, sigma2 * torch.eye(d))
#     samples = dist.sample((num_samples,))
#     labels = torch.ones(num_samples)  # Single class for single Gaussian
#     return samples, labels

# ==============================================================
# 1. Source: single Gaussian
# 2. Target: mixture of two Gaussians (main + side)
# ==============================================================

# Import enhanced networks - create a simple implementation
class EnhancedNoisePredictor(nn.Module):
    """
    Enhanced Noise Predictor with time embedding
    """
    def __init__(self, input_dim=2, hidden_dim=384, time_embed_dim=128, num_layers=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Main network
        layers = []
        in_dim = input_dim + time_embed_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)
        
        # Concatenate x and time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        return self.network(x_t)

def generate_source(mu, sigma2, num_samples, d=2, device="cpu"):
    """
    Generate single-cluster source samples.
    """
    dist = torch.distributions.MultivariateNormal(mu, sigma2 * torch.eye(d, device=device))
    samples = dist.sample((num_samples,))
    labels = torch.ones(num_samples)  # label +1 for source
    return samples.to(device), labels.to(device)


def generate_target(mu_main, mu_side, sigma2_main, sigma2_side, w_main, num_samples, d=2, device="cpu"):
    """
    Generate target mixture: main cluster + side cluster.
    """
    n_main = int(num_samples * w_main)
    n_side = num_samples - n_main

    dist_main = torch.distributions.MultivariateNormal(mu_main, sigma2_main * torch.eye(d, device=device))
    dist_side = torch.distributions.MultivariateNormal(mu_side, sigma2_side * torch.eye(d, device=device))
    
    samples_main = dist_main.sample((n_main,))
    samples_side = dist_side.sample((n_side,))
    samples = torch.cat([samples_main, samples_side], dim=0)
    
    # Label main=+1, side=-1 (optional)
    labels = torch.cat([torch.ones(n_main), -torch.ones(n_side)])
    return samples.to(device), labels.to(device)


def compute_target_likelihood(samples, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device="cpu"):
    """
    Compute log-likelihood of samples under the target Gaussian mixture distribution.
    """
    # Create mixture components
    dist_main = torch.distributions.MultivariateNormal(mu_main, sigma2_main * torch.eye(2, device=device))
    dist_side = torch.distributions.MultivariateNormal(mu_side, sigma2_side * torch.eye(2, device=device))
    
    # Compute log probabilities
    log_prob_main = dist_main.log_prob(samples)
    log_prob_side = dist_side.log_prob(samples)
    
    # Mixture log-likelihood: log(w1 * p1 + w2 * p2) = logsumexp(log(w1) + log(p1), log(w2) + log(p2))
    w_main_tensor = torch.tensor(w_main, device=device)
    w_side_tensor = torch.tensor(1 - w_main, device=device)
    
    log_mixture = torch.logsumexp(
        torch.stack([
            torch.log(w_main_tensor) + log_prob_main,
            torch.log(w_side_tensor) + log_prob_side
        ]), dim=0
    )
    
    return log_mixture


def compute_source_likelihood(samples, mu_source, sigma2_source, device="cpu"):
    """
    Compute log-likelihood of samples under the source Gaussian distribution.
    """
    dist_source = torch.distributions.MultivariateNormal(mu_source, sigma2_source * torch.eye(2, device=device))
    return dist_source.log_prob(samples)


def compute_density_ratio(samples, mu_source, sigma2_source, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device="cpu"):
    """
    Compute the analytical density ratio q(x)/p(x) for Gaussian distributions.
    
    Args:
        samples: Input samples [N, 2]
        mu_source: Source distribution mean [2]
        sigma2_source: Source distribution variance
        mu_main: Target main cluster mean [2]
        mu_side: Target side cluster mean [2]
        sigma2_main: Target main cluster variance
        sigma2_side: Target side cluster variance
        w_main: Weight of main cluster
        device: Device
    
    Returns:
        density_ratio: q(x)/p(x) for each sample [N]
    """
    # Source distribution: p(x) = N(mu_source, sigma2_source * I)
    dist_source = torch.distributions.MultivariateNormal(mu_source, sigma2_source * torch.eye(2, device=device))
    
    # Target distribution: q(x) = w_main * N(mu_main, sigma2_main * I) + w_side * N(mu_side, sigma2_side * I)
    dist_main = torch.distributions.MultivariateNormal(mu_main, sigma2_main * torch.eye(2, device=device))
    dist_side = torch.distributions.MultivariateNormal(mu_side, sigma2_side * torch.eye(2, device=device))
    
    # Compute densities
    p_density = torch.exp(dist_source.log_prob(samples))  # p(x)
    q_main_density = torch.exp(dist_main.log_prob(samples))  # N(mu_main, sigma2_main)
    q_side_density = torch.exp(dist_side.log_prob(samples))  # N(mu_side, sigma2_side)
    
    # Mixture density: q(x) = w_main * q_main + w_side * q_side
    w_side = 1 - w_main
    q_density = w_main * q_main_density + w_side * q_side_density
    
    # Density ratio: q(x)/p(x)
    density_ratio = q_density / p_density
    
    return density_ratio

def generate_source_cosine(num_samples, device="cpu"):
    """
    Generate uniform source samples on [-1, 1]^2.
    Source distribution p(x) = 1/4 for x ∈ [-1, 1]^2
    """
    # Uniform distribution on [-1, 1]^2
    samples = torch.rand(num_samples, 2, device=device) * 2 - 1  # Scale to [-1, 1]
    labels = torch.ones(num_samples, device=device)  # label +1 for source
    return samples, labels


def generate_target_cosine(num_samples, a=5, device="cpu"):
    """
    Generate target samples with ratio range [1/a, a].
    q(x)/p(x) = c * (1 + b * cos(pi x1) * cos(pi x2)),
    where c and b are computed from a.

    Args:
        num_samples: number of samples to generate
        a: ratio range parameter (a >= 1), ensures q/p ∈ [1/a, a]
        device: torch device ("cpu" or "cuda")

    Returns:
        samples: [num_samples, 2] tensor in [-1, 1]^2
        labels: +1 tensor labels (for target domain)
    """
    assert a >= 1, "Parameter a must be >= 1"

    # Compute scaling coefficients from target ratio range [1/a, a]
    r_min, r_max = 1 / a, a
    c = 0.5 * (r_max + r_min)
    b = (r_max - r_min) / (r_max + r_min)

    # Vectorized rejection sampling
    N_prop = int(num_samples * 3)  # oversample factor
    x = 2 * torch.rand(N_prop, 2, device=device) - 1
    ratio = c * (1 + b * torch.cos(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1]))

    # Normalize acceptance probabilities to [0,1]
    accept_prob = ratio / ratio.max()
    u = torch.rand_like(accept_prob)
    accepted = x[u < accept_prob]
    print(f"Accepted samples: {len(accepted)}")

    # Retry if not enough samples accepted
    if len(accepted) < num_samples:
        return generate_target_cosine(num_samples, a=a, device=device)

    samples = accepted[:num_samples]
    labels = torch.ones(num_samples, device=device)
    return samples, labels


# ==============================================================
# 2. Define parameters for cosine modulation example
# ==============================================================

d = 2
a = 5  # Cosine modulation parameter, controls density ratio bounds [1/a, a] = [0.2, 5]

# Support set: S = [-1, 1]^2 (bounded)
# Source: p(x) = 1/4 (uniform on [-1, 1]^2)
# Target: q(x) = 1/4 * (1 + a * cos(πx1) * cos(πx2))
# Density ratio: q(x)/p(x) = 1 + a * cos(πx1) * cos(πx2) ∈ [1-a, 1+a]

m = 10000
device = args.device

# ==============================================================
# 3. Generate samples
# ==============================================================

# Gaussian distributions for likelihood calculation
# Source: single Gaussian at (1, 0)
mu_source = torch.tensor([1.0, 0.0], device=device)
sigma2_source = 0.5
source_data, source_labels = generate_source(mu_source, sigma2_source, m, d=2, device=device)

# Target: mixture of two Gaussians
# Main cluster close to source at (1, 1)
mu_main = torch.tensor([1.0, 1.0], device=device)
# Side cluster at (-0.5, -0.5)
mu_side = torch.tensor([-0.5, -0.5], device=device)
sigma2_main = 0.3
sigma2_side = 0.2
w_main = 0.7  # 70% main cluster, 30% side cluster

target_data_all_n, target_labels_all_n = [], []
target_data, target_labels = generate_target(mu_main, mu_side, sigma2_main, sigma2_side, w_main, args.n, d=2, device=device)
target_data_all_n.append(target_data)
target_labels_all_n.append(target_labels)
# pick one for training
n_target_index = 0
target_labels_ = target_labels_all_n[n_target_index]
target_data_ = target_data_all_n[n_target_index]

# Generate 5000 target samples for plotting (more samples for better heatmap)
target_data_plot, target_labels_plot = generate_target(mu_main, mu_side, sigma2_main, sigma2_side, w_main, 5000, d=2, device=device)
source_data_plot, source_labels_plot = generate_source(mu_source, sigma2_source, 5000, d=2, device=device)


# -----------------------------------------------------------------------
# 2. Define Resampling Methods (from animation.py)
# -----------------------------------------------------------------------

def sinkhorn_barycentric_projection(X, w, Y=None, eps=0.2, max_iter=300, tol=1e-9):
    """
    对离散源分布 sum_i w_i δ_{X_i} 到等权 1/m sum_j δ_{Y_j} 做熵正则OT，
    返回重心投影后的等权粒子 X_out（大小与 Y 相同；若 Y=None，则取 Y=X，输出与输入个数一致）。

    参数:
      - X: (n,d) 源点
      - w: (n,)  源权重，和为1
      - Y: (m,d) 目标锚点；None 时取 X.copy()
      - eps: 熵正则系数（越小越接近真OT，越大越平滑）
    返回:
      - X_out: (m,d) OT重采样后的等权输出点
    """
    X = np.asarray(X)
    w = np.asarray(w)
    n, d = X.shape
    if Y is None:
        Y = X.copy()
    Y = np.asarray(Y)
    m = Y.shape[0]

    # 代价矩阵 C_ij = ||X_i - Y_j||^2（稳定写法）
    X2 = np.sum(X**2, axis=1, keepdims=True)    # (n,1)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T  # (1,m)
    C = X2 + Y2 - 2 * X @ Y.T                   # (n,m)

    # 熵核
    K = np.exp(-C / max(eps, 1e-6)) + 1e-300

    # 目标边缘（等权）
    u = np.ones(m) / m

    # Sinkhorn 缩放
    a = np.ones(n)
    b = np.ones(m)
    for _ in range(max_iter):
        a_old, b_old = a, b
        Kb = K @ b
        Kb = np.maximum(Kb, 1e-300)
        a = w / Kb

        KTa = K.T @ a
        KTa = np.maximum(KTa, 1e-300)
        b = u / KTa

        if np.max(np.abs(a - a_old)) < tol and np.max(np.abs(b - b_old)) < tol:
            break

    # 耦合矩阵
    T = (a[:, None] * K) * b[None, :]   # (n,m)，列和≈1/m

    # 重心投影（barycentric projection）：每列 j 的新位置为 m * sum_i T_ij X_i
    X_out = (m * (X.T @ T)).T  # (m,d)
    return X_out


def multinomial_resampling(x, weights):
    """
    多项式重采样
    """
    N = x.shape[0]
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    idx = np.random.choice(np.arange(N), size=N, p=weights)
    return x[idx]


def ot_resampling(x, weights, eps=0.2, max_iter=300, tol=1e-9):
    """
    最优传输重采样
    """
    try:
        x_np = x.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        x_resampled = sinkhorn_barycentric_projection(x_np, weights_np, Y=None, eps=eps, max_iter=max_iter, tol=tol)
        return torch.tensor(x_resampled, dtype=x.dtype, device=x.device)
    except Exception as e:
        print(f"OT resampling failed, falling back to multinomial: {e}")
        return multinomial_resampling(x, weights)


# ==============================================================
# Visualization functions for particle movement during resampling
# ==============================================================
# Set resampling steps based on strategy
if args.resampling_strategy == 'every_step':
    resampling_steps = [i for i in range(1, 26)]  # Every step: 1,2,3,...,25
elif args.resampling_strategy == 'sparse':
    resampling_steps = [2, 4, 8, 16]  # Sparse: 2,4,8,16
else:
    resampling_steps = [8, 15, 22]  # Default fallback

print(f"Resampling strategy: {args.resampling_strategy}")
print(f"Resampling steps: {resampling_steps}")

# Create results directory with resampling strategy
results_dir = f"results/resampling_gaussian_{args.resampling_strategy}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

def create_resampling_animation(particle_paths, method_name, resampling_steps=resampling_steps, save_path=None):
    """
    创建重采样过程的动画GIF
    
    Args:
        particle_paths: 粒子轨迹列表，每个元素是 [N, 2] 的粒子位置
        method_name: 方法名称
        resampling_steps: 重采样步骤列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np
    
    # 转换为numpy数组
    if isinstance(particle_paths[0], torch.Tensor):
        particle_paths = [path.detach().cpu().numpy() for path in particle_paths]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制目标分布背景
    def draw_background(ax):
        # 创建网格
        xg, yg = np.mgrid[-3:3:0.1, -3:3:0.1]
        
        # 计算目标分布密度 (余弦调制分布)
        cos_term = np.cos(np.pi * xg) * np.cos(np.pi * yg)
        target_density = 0.25 * (1 + 5 * cos_term)  # a=5
        
        # 绘制等高线
        ax.contour(xg, yg, target_density, levels=8, cmap='Blues', alpha=0.6, linewidths=1)
        ax.contourf(xg, yg, target_density, levels=8, cmap='Blues', alpha=0.3)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('X₁', fontsize=12)
        ax.set_ylabel('X₂', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # 初始化背景
    draw_background(ax1)
    draw_background(ax2)
    
    # 选择关键帧（每3步一帧 + 重采样步骤）
    frames_to_plot = list(range(0, len(particle_paths), 3))
    
    # 添加重采样步骤
    if resampling_steps:
        for step in resampling_steps:
            if step < len(particle_paths) and step not in frames_to_plot:
                frames_to_plot.append(step)
    
    # 确保最后一帧被包含
    if len(particle_paths) - 1 not in frames_to_plot:
        frames_to_plot.append(len(particle_paths) - 1)
    
    # 排序帧列表
    frames_to_plot = sorted(frames_to_plot)
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # 重新绘制背景
        draw_background(ax1)
        draw_background(ax2)
        
        # 获取当前帧的粒子位置
        current_particles = particle_paths[frame_idx]
        
        # 左图：粒子分布和轨迹
        # 绘制轨迹（从开始到当前帧）
        for i in range(min(20, len(current_particles))):  # 只显示前20个粒子的轨迹
            trajectory = np.array([particle_paths[k][i] for k in range(0, frame_idx + 1, 2)])
            if len(trajectory) > 1:
                ax1.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, linewidth=0.8, color='gray')
        
        # 绘制当前粒子位置
        ax1.scatter(current_particles[:, 0], current_particles[:, 1], s=8, alpha=0.7, c='red')
        ax1.set_title(f'{method_name} - Step {frame_idx}', fontsize=14, pad=15)
        
        # 标记重采样步骤
        if resampling_steps and frame_idx in resampling_steps:
            ax1.text(-2.5, 2.5, f"RESAMPLE @ t={frame_idx}", fontsize=12, color='red', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        elif resampling_steps and frame_idx > max(resampling_steps):
            ax1.text(-2.5, 2.5, f"After resample (t={frame_idx})", fontsize=10, color='darkred', weight='bold')
        
        # 右图：粒子密度热图
        from scipy.stats import gaussian_kde
        
        if len(current_particles) > 1:
            try:
                # 计算KDE
                kde = gaussian_kde(current_particles.T, bw_method=0.3)
                
                # 创建网格
                x_min, x_max = -3, 3
                y_min, y_max = -3, 3
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
                grid_coords = np.vstack([xx.ravel(), yy.ravel()])
                
                # 计算密度
                density = kde(grid_coords).reshape(xx.shape)
                
                # 绘制密度热图
                im = ax2.contourf(xx, yy, density, levels=15, cmap='viridis', alpha=0.8)
                ax2.scatter(current_particles[:, 0], current_particles[:, 1], s=4, alpha=0.6, 
                           c='white', edgecolors='black', linewidth=0.5)
                ax2.set_title(f'Particle Density - Step {frame_idx}', fontsize=14, pad=15)
                
            except Exception as e:
                ax2.scatter(current_particles[:, 0], current_particles[:, 1], s=8, alpha=0.7, c='red')
                ax2.set_title(f'Particle Density - Step {frame_idx}', fontsize=14, pad=15)
        else:
            ax2.scatter(current_particles[:, 0], current_particles[:, 1], s=8, alpha=0.7, c='red')
            ax2.set_title(f'Particle Density - Step {frame_idx}', fontsize=14, pad=15)
        
        # 设置总标题
        fig.suptitle(f'{method_name} - Step {frame_idx}', fontsize=16)
        
        return []
    
    # 创建动画
    print(f"Creating animation for {method_name}...")
    try:
        ani = FuncAnimation(fig, update, frames=frames_to_plot, interval=800, blit=False, repeat=True)
        writer = PillowWriter(fps=1.2)
        ani.save(save_path, writer=writer)
        plt.close()
        print(f"✅ Animation saved: {save_path}")
    except Exception as e:
        print(f"Animation failed: {e}")
        # 保存最终静态图作为备选
        static_path = save_path.replace('.gif', '_final.png')
        update(len(particle_paths) - 1)
        plt.savefig(static_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Static image saved: {static_path}")


def visualize_particle_movement(x_samples, step, method_name, resampling_steps=None, save_path=None):
    """
    可视化重采样过程中粒子的移动（单帧）
    
    Args:
        x_samples: 当前步骤的粒子位置 [N, 2]
        step: 当前扩散步骤
        method_name: 方法名称
        resampling_steps: 重采样步骤列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 转换为numpy数组
    if isinstance(x_samples, torch.Tensor):
        x_samples = x_samples.detach().cpu().numpy()
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：粒子分布
    ax1.scatter(x_samples[:, 0], x_samples[:, 1], s=8, alpha=0.7, c='blue')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_title(f'{method_name} - Step {step}')
    ax1.grid(True, alpha=0.3)
    
    # 标记重采样步骤
    if resampling_steps and step in resampling_steps:
        ax1.text(-2.5, 2.5, f"RESAMPLE @ t={step}", fontsize=12, color='red', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 右图：粒子密度热图
    from scipy.stats import gaussian_kde
    
    if len(x_samples) > 1:
        try:
            # 计算KDE
            kde = gaussian_kde(x_samples.T, bw_method=0.3)
            
            # 创建网格
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])
            
            # 计算密度
            density = kde(grid_coords).reshape(xx.shape)
            
            # 绘制密度热图
            im = ax2.contourf(xx, yy, density, levels=20, cmap='viridis', alpha=0.8)
            ax2.scatter(x_samples[:, 0], x_samples[:, 1], s=4, alpha=0.6, c='white', edgecolors='black', linewidth=0.5)
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_xlabel('X₁')
            ax2.set_ylabel('X₂')
            ax2.set_title(f'Particle Density - Step {step}')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax2, label='Density')
            
        except Exception as e:
            ax2.text(0.5, 0.5, f'KDE failed: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Particle Density - Step {step}')
    else:
        ax2.text(0.5, 0.5, 'Not enough samples for KDE', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Particle Density - Step {step}')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Particle movement visualization saved: {save_path}")
    
    plt.close(fig)


def log_resampling_stats(x_samples, weights, step, method_name):
    """
    记录重采样统计信息
    
    Args:
        x_samples: 粒子位置 [N, 2]
        weights: 权重 [N]
        step: 当前步骤
        method_name: 方法名称
    """
    if isinstance(x_samples, torch.Tensor):
        x_samples = x_samples.detach().cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # 计算统计信息
    x_mean = np.mean(x_samples, axis=0)
    x_std = np.std(x_samples, axis=0)
    x_range = [np.min(x_samples, axis=0), np.max(x_samples, axis=0)]
    
    weight_mean = np.mean(weights)
    weight_std = np.std(weights)
    weight_min = np.min(weights)
    weight_max = np.max(weights)
    
    # 检查粒子多样性
    unique_particles = len(np.unique(np.round(x_samples, 3), axis=0))
    diversity_ratio = unique_particles / len(x_samples)
    
    print(f"[{method_name}] Step {step}:")
    print(f"  Particles: {len(x_samples)}, Unique: {unique_particles} ({diversity_ratio:.1%})")
    print(f"  Position: mean=({x_mean[0]:.3f}, {x_mean[1]:.3f}), std=({x_std[0]:.3f}, {x_std[1]:.3f})")
    print(f"  Weights: mean={weight_mean:.6f}, std={weight_std:.6f}, range=[{weight_min:.6f}, {weight_max:.6f}]")
    
    if diversity_ratio < 0.1:
        print(f"  ⚠️  WARNING: Very low particle diversity!")
    elif diversity_ratio < 0.5:
        print(f"  ⚠️  Note: Low particle diversity")


# -----------------------------------------------------------------------
# 3. Define the Networks: Noise Predictor, Guidance Network and Classifier
# -----------------------------------------------------------------------

# Use EnhancedNoisePredictor instead of the original NoisePredictor
# class NoisePredictor(nn.Module):
#     """
#     A small MLP that takes (x_t, t) and predicts noise.
#     We feed t as an extra input dimension.
#     """
#     def __init__(self, input_dim=d, hidden_dim=[512,512,512,512,256]):
#         super().__init__()
#         output_dim = input_dim
#         in_dim = input_dim + 1  # +1 for t
#         layers = []
#         for h_dim in hidden_dim:
#             layers.append(nn.Linear(in_dim, h_dim))
#             layers.append(nn.SiLU())
#             in_dim = h_dim
#         layers.append(nn.Linear(in_dim, output_dim))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x, t):
#         # t is [batch], expand to [batch,1] then cat
#         #t_emb = t.view(-1, 1).expand(-1, x.shape[1])
#         if len(t.shape) == 1:
#             t = t.unsqueeze(1)
#         inp = torch.cat([x, t], dim=1)
#         return self.model(inp)

# Use EnhancedNoisePredictor as the main NoisePredictor
class NoisePredictor(EnhancedNoisePredictor):
    """Enhanced Noise Predictor for cosine modulation experiments"""
    def __init__(self, input_dim=d, hidden_dim=384, time_embed_dim=128, num_layers=5):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, 
                        time_embed_dim=time_embed_dim, num_layers=num_layers)

#noise_predictor = NoisePredictor()
#torchsummary.summary(noise_predictor, input_size=[(2,),(1,)])


class GuidanceNetwork(nn.Module):
    """
    4-layer MLP with 512 hidden units and SiLU activation function
    Only x1 and x2 as input dimensions
    """
    def __init__(self, input_dim=d, hidden_dim=[512,512,512,512]):
        super().__init__()
        in_dim = input_dim+1 # +1 for t
        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
    def forward(self, x):
        return self.softplus(self.model(x))
#guidance_network = GuidanceNetwork()
#torchsummary.summary(guidance_network, input_size=[(2,)])

class Classifier(nn.Module):
    """
    The is no Information on how the classifier should look like in the Paper, so I defined it like the Guidance Network
    4-layer MLP with 512 hidden units and SiLU activation function and a sigmoid output layer
    Only x1 and x2 as input dimensions
    """
    def __init__(self, input_dim=d, hidden_dim=[512,512,512,512]):
        super().__init__()
        in_dim = input_dim
        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)


class TimeModulatedLayer(nn.Module):
    """
    时间调制层：使用时间嵌入来调制空间特征的处理
    """
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        
        # 主要的空间特征处理
        self.spatial_transform = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU()
        )
        
        # 时间调制参数生成 (scale and shift)
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim * 2)  # 生成 scale 和 shift
        )
        
    def forward(self, x, t_emb):
        """
        x: [batch_size, hidden_dim] - 空间特征
        t_emb: [batch_size, hidden_dim] - 时间嵌入
        """
        # 1. 空间特征处理
        h = self.spatial_transform(x)
        
        # 2. 时间调制参数
        modulation = self.time_modulation(t_emb)
        scale, shift = torch.chunk(modulation, 2, dim=-1)
        
        # 3. 应用时间调制 (类似 FiLM: Feature-wise Linear Modulation)
        h = h * (1 + scale) + shift
        
        return h


class TimeDependentClassifier(nn.Module):
    """
    时间相关域分类器/引导网络，专门设计用于处理 (x_t, t) 输入
    核心思想：
    1. 时间步 t 通过正弦/余弦嵌入映射到高维空间
    2. 时间嵌入通过门控/调制机制影响空间特征处理
    3. 多层次融合时间和空间信息
    
    可以用于：
    1. 时间相关域分类器: c_ω(x_t, t) (输出概率，可选sigmoid)
    2. 时间相关引导网络: h_ψ(x_t, t) (输出引导值)
    """
    def __init__(self, input_dim=d, hidden_dim=512, time_embed_dim=128, num_layers=4, use_sigmoid=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.use_sigmoid = use_sigmoid
        
        # 1. 时间步嵌入网络 (正弦/余弦位置编码 + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. 空间特征初始处理
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 3. 时间调制层 (Time-modulated layers)
        self.modulated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.modulated_layers.append(
                TimeModulatedLayer(hidden_dim, hidden_dim)
            )
        
        # 4. 输出层
        output_layers = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        ]
        
        # 根据use_sigmoid参数决定是否添加sigmoid
        if use_sigmoid:
            output_layers.append(nn.Sigmoid())
            
        self.output_layer = nn.Sequential(*output_layers)
        
    def get_time_embedding(self, t):
        """
        使用正弦/余弦位置编码将时间步映射到高维空间
        t: [batch_size] 或 [batch_size, 1]
        返回: [batch_size, time_embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [batch_size, 1]
        
        # 创建不同频率的正弦/余弦编码
        half_dim = self.time_embed_dim // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t * emb.unsqueeze(0)  # [batch_size, half_dim]
        
        # 拼接正弦和余弦
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [batch_size, time_embed_dim]
        return emb
    
    def forward(self, x_t, t=None):
        """
        x_t: [batch_size, input_dim] 或 [batch_size, input_dim+1] - 噪声数据或拼接后的数据
        t: [batch_size] 或 [batch_size, 1] 或 None - 时间步
        """
        # 处理输入：如果t为None，说明x_t已经包含了时间信息
        if t is None:
            # x_t是拼接后的输入 [batch_size, input_dim+1]
            x = x_t[:, :self.input_dim]  # 提取空间部分
            t = x_t[:, self.input_dim:]  # 提取时间部分
            if t.shape[1] == 1:
                t = t.squeeze(1)  # [batch_size]
        else:
            # x_t是纯空间输入，t是分离的时间
            x = x_t
        
        # 1. 时间步嵌入
        t_emb = self.get_time_embedding(t)  # [batch_size, time_embed_dim]
        t_emb = self.time_embed(t_emb)  # [batch_size, hidden_dim]
        
        # 2. 空间特征编码
        h = self.spatial_encoder(x)  # [batch_size, hidden_dim]
        
        # 3. 时间调制的特征处理
        for layer in self.modulated_layers:
            h = layer(h, t_emb)
        
        # 4. 输出
        out = self.output_layer(h)  # [batch_size, 1]
        return out


# 为了向后兼容，提供TimeDependentGuidanceNetwork别名
class TimeDependentGuidanceNetwork(TimeDependentClassifier):
    """
    时间相关引导网络 - TimeDependentClassifier的别名，不使用sigmoid
    """
    def __init__(self, input_dim=d, hidden_dim=512, time_embed_dim=128, num_layers=4):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, 
                        time_embed_dim=time_embed_dim, num_layers=num_layers, 
                        use_sigmoid=False)

#classifier = Classifier()
#torchsummary.summary(classifier, input_size=[(2,)])

# -----------------------------------------------------------------------
# 3. Define the Noise Schedule from the Paper DPM Solver
# -----------------------------------------------------------------------
class NoiseScheduleVP:
    """
    Minimal version that lets us get alpha_t, sigma_t, etc.
    For simplicity, let's do a 'linear' schedule (as an example).
    """
    def __init__(self, beta_0=0.1, beta_1=20., dtype=torch.float32):
        self.schedule = 'linear'
        self.T = 1.0
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.dtype = dtype

    def marginal_log_mean_coeff(self, t):
        # log(alpha_t)
        # I asumed that they made a mistake in the paper and the formula should be like this
        # For linear VPSDE: alpha_t = exp(-0.5 * (beta_0 * t + 0.5*(beta_1-beta_0)*t^2))
        return -0.5*(self.beta_0*t + 0.5*(self.beta_1-self.beta_0)*t**2)

    def marginal_lambda(self, t):
        # Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].

        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2.*self.marginal_log_mean_coeff(t)))

    def inverse_lambda(self, lamb):
        # Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)


noise_schedule = NoiseScheduleVP(beta_0=0.1, beta_1=20.0)

# -----------------------------------------------------------------------
# 3. Train the Noise Predictor - Based on the work: https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py#L30
# -----------------------------------------------------------------------
def train_diffusion_noise_prediction(
    model, data, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'
):
    """
    Minimal training loop for noise-prediction on 2D data.
    q(x_t|x_0) ~ alpha_t * x_0 + sigma_t * eps, with t ~ Uniform(0,1).
    Loss = MSE(predicted_noise, true_noise).
    """
    if batch_size > data.size(0):
        batch_size = data.size(0)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        for batch in loader:
            x0 = batch[0].to(device)    # shape [batch, 2]
            t_ = torch.rand(x0.size(0), device=device)  # uniform in [0,1]
            a_t = noise_schedule.marginal_alpha(t_).unsqueeze(1)
            s_t = noise_schedule.marginal_std(t_).unsqueeze(1)

            eps = torch.randn_like(x0)
            x_t = a_t * x0 + s_t * eps  # forward diffusion

            # Predict the noise
            eps_pred = model(x_t, t_)

            loss = mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion] epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")

    model.to('cpu')


def train_diffusion_noise_prediction_importance_sampling(
    model, data, classifier_time_dependent, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'
):
    """
    Importance Sampling version of noise-prediction training.
    Loss = h_omega(x_t, t) * || s_theta(alpha(t)epsilon + sigma(t)X1, t) + epsilon / alpha(t) ||^2_2
    where h_omega(x_t, t) is the ratio q_t/p_t determined by the time-dependent classifier.
    """
    if batch_size > data.size(0):
        batch_size = data.size(0)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)
    classifier_time_dependent.to(device)
    classifier.eval()  # Time-dependent classifier is assumed pre-trained/frozen

    for epoch in range(n_epochs):
        for batch in loader:
            x0 = batch[0].to(device)    # shape [batch, 2]
            t_ = torch.rand(x0.size(0), device=device)  # uniform in [0,1]
            a_t = noise_schedule.marginal_alpha(t_).unsqueeze(1)
            s_t = noise_schedule.marginal_std(t_).unsqueeze(1)

            eps = torch.randn_like(x0)
            x_t = a_t * x0 + s_t * eps  # forward diffusion

            # Predict the noise
            eps_pred = model(x_t, t_)

            # Compute the base loss: || s_theta(...) + epsilon / alpha(t) ||^2_2
            base_loss = mse_loss(eps_pred, eps)
            
            # Get the importance ratio from time-dependent classifier: h_omega(x_t, t)
            with torch.no_grad():
                classifier_out = classifier_time_dependent(x_t, t_)
                
                # Clamp classifier output to prevent extreme ratios
                classifier_out_clamped = torch.clamp(classifier_out, min=1e-8, max=1-1e-8)
                ratio = (1 - classifier_out_clamped) / classifier_out_clamped  # h_omega(x_t, t) = q_t/p_t
                
                # Clamp the final ratio to reasonable range [1e-8, 1e8]
                ratio = torch.clamp(ratio, min=1e-8, max=1e8)

            # Importance sampling loss: h_omega(x_t, t) * base_loss
            loss = torch.mean(ratio * base_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion IS] epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")
            
        # Memory cleanup every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.cuda.empty_cache() if device != 'cpu' else None

    model.to('cpu')
    classifier.to('cpu')


def train_diffusion_noise_prediction_importance_sampling_t_power(
    model, data, classifier_time_dependent, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'
):
    """
    Importance Sampling version with ratio raised to power t.
    Loss = h_omega(x_t, t)^t * || s_theta(alpha(t)epsilon + sigma(t)X1, t) + epsilon / alpha(t) ||^2_2
    where h_omega(x_t, t) is the ratio q_t/p_t determined by the time-dependent classifier.
    """
    if batch_size > data.size(0):
        batch_size = data.size(0)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)

    classifier.eval()  # Time-dependent classifier is assumed pre-trained/frozen

    for epoch in range(n_epochs):
        for batch in loader:
            x0 = batch[0].to(device)    # shape [batch, 2]
            t_ = torch.rand(x0.size(0), device=device)  # uniform in [0,1]
            a_t = noise_schedule.marginal_alpha(t_).unsqueeze(1)
            s_t = noise_schedule.marginal_std(t_).unsqueeze(1)

            eps = torch.randn_like(x0)
            x_t = a_t * x0 + s_t * eps  # forward diffusion

            # Predict the noise
            eps_pred = model(x_t, t_)

            # Compute the base loss: || s_theta(...) + epsilon / alpha(t) ||^2_2
            base_loss = mse_loss(eps_pred, eps)
            
            # Get the importance ratio from time-dependent classifier: h_omega(x_t, t)
            with torch.no_grad():
                classifier_out = classifier_time_dependent(x_t, t_)
                
                # Clamp classifier output to prevent extreme ratios
                classifier_out_clamped = torch.clamp(classifier_out, min=1e-8, max=1-1e-8)
                ratio = (1 - classifier_out_clamped) / classifier_out_clamped  # h_omega(x_t, t) = q_t/p_t
                
                # Clamp the final ratio to reasonable range [1e-8, 1e8]
                ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                
                # Apply t power: ratio^t
                ratio_t_power = torch.pow(ratio, t_.unsqueeze(1))

            # Importance sampling loss with t power: h_omega(x_t, t)^t * base_loss
            loss = torch.mean(ratio_t_power * base_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion IS t-power] epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")
            
        # Memory cleanup every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.cuda.empty_cache() if device != 'cpu' else None

    model.to('cpu')
    classifier.to('cpu')


def train_diffusion_noise_prediction_diff_tuning(
    model, source_data, target_data, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'
):
    """
    Diff-Tuning: Combined Knowledge Retention and Knowledge Reconsolidation
    
    Loss = L_retention(θ) + L_adaptation(θ)
    
    L_retention(θ) = E_{t,ε,x̂₀ˢ~X̂ˢ} [ξ(t) ||ε - f_θ(√(α_t)x̂₀ˢ + √(1-α_t)ε, t)||²]
    L_adaptation(θ) = E_{t,ε,x₀~X} [ψ(t) ||ε - f_θ(√(α_t)x₀ + √(1-α_t)ε, t)||²]
    
    where:
    - ξ(t): retention coefficient (monotonically decreasing)
    - ψ(t): reconsolidation coefficient (monotonically increasing)
    - X̂ˢ: augmented source dataset (pre-sampled from pre-trained model)
    - X: target domain dataset
    
    Enhanced version: Uses up to 3x more source samples per epoch than target samples
    """
    # Calculate the maximum number of source samples to use per epoch
    target_size = target_data.size(0)
    source_size = source_data.size(0)
    max_source_per_epoch = min(3 * target_size, source_size)
    
    print(f"Diff-Tuning: Using up to {max_source_per_epoch} source samples per epoch (target: {target_size})")
    
    if batch_size > min(max_source_per_epoch, target_size):
        batch_size = min(max_source_per_epoch, target_size)
    
    # Create datasets
    source_dataset = TensorDataset(source_data)
    target_dataset = TensorDataset(target_data)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        # Create iterators for both datasets
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Calculate number of batches for target (this determines the epoch length)
        n_target_batches = len(target_loader)
        
        # Calculate how many source batches we can use (up to 3x target batches)
        n_source_batches = min(3 * n_target_batches, len(source_loader))
        
        # Process all target batches
        for batch_idx in range(n_target_batches):
            # Get target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            x0_target = target_batch[0].to(device)      # shape [batch, 2]
            
            # Sample time steps for target
            t_target = torch.rand(x0_target.size(0), device=device)  # uniform in [0,1]
            
            # Forward diffusion for target data
            a_t_target = noise_schedule.marginal_alpha(t_target).unsqueeze(1)
            s_t_target = noise_schedule.marginal_std(t_target).unsqueeze(1)
            eps_target = torch.randn_like(x0_target)
            x_t_target = a_t_target * x0_target + s_t_target * eps_target
            
            # Predict noise for target
            eps_pred_target = model(x_t_target, t_target)
            
            # Compute base loss for target
            base_loss_target = mse_loss(eps_pred_target, eps_target)
            
            # Compute reconsolidation coefficient ψ(t) (monotonically increasing)
            # ψ(t) = t (simple linear increasing function)
            reconsolidation_coeff = t_target  # shape [batch]
            
            # Compute adaptation loss
            adaptation_loss = torch.mean(reconsolidation_coeff * base_loss_target)
            
            # Process source data (up to 3 batches per target batch)
            source_batches_processed = 0
            max_source_batches_per_target = min(3, n_source_batches // n_target_batches)
            
            retention_losses = []
            while source_batches_processed < max_source_batches_per_target and source_batches_processed < n_source_batches:
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)
                
                x0_source = source_batch[0].to(device)    # shape [batch, 2]
                
                # Sample time steps for source
                t_source = torch.rand(x0_source.size(0), device=device)  # uniform in [0,1]
                
                # Forward diffusion for source data
                a_t_source = noise_schedule.marginal_alpha(t_source).unsqueeze(1)
                s_t_source = noise_schedule.marginal_std(t_source).unsqueeze(1)
                eps_source = torch.randn_like(x0_source)
                x_t_source = a_t_source * x0_source + s_t_source * eps_source
                
                # Predict noise for source
                eps_pred_source = model(x_t_source, t_source)
                
                # Compute base loss for source
                base_loss_source = mse_loss(eps_pred_source, eps_source)
                
                # Compute retention coefficient ξ(t) (monotonically decreasing)
                # ξ(t) = 1 - t (simple linear decreasing function)
                retention_coeff = 1.0 - t_source  # shape [batch]
                
                # Compute retention loss
                retention_loss = torch.mean(retention_coeff * base_loss_source)
                retention_losses.append(retention_loss)
                
                source_batches_processed += 1
            
            # Average retention loss over all source batches processed
            if retention_losses:
                avg_retention_loss = torch.stack(retention_losses).mean()
            else:
                avg_retention_loss = torch.tensor(0.0, device=device)
            
            # Total loss: L_retention(θ) + L_adaptation(θ)
            total_loss = avg_retention_loss + adaptation_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion Diff-Tuning] epoch {epoch+1}/{n_epochs}, "
                  f"total_loss={total_loss.item():.4f}, "
                  f"retention_loss={avg_retention_loss.item():.4f}, "
                  f"adaptation_loss={adaptation_loss.item():.4f}")
            
        # Memory cleanup every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.cuda.empty_cache() if device != 'cpu' else None

    model.to('cpu')


def train_diffusion_noise_prediction_diff_tuning_ratio(
    model, source_data, target_data, classifier, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'
):
    """
    Diff-Tuning with Ratio: Combined Knowledge Retention and Knowledge Reconsolidation
    
    Loss = L_retention(θ) + L_adaptation(θ)
    
    L_retention(θ) = E_{t,ε,x̂₀ˢ~X̂ˢ} [ratio^t * ||ε - f_θ(√(α_t)x̂₀ˢ + √(1-α_t)ε, t)||²]
    L_adaptation(θ) = E_{t,ε,x₀~X} [1 * ||ε - f_θ(√(α_t)x₀ + √(1-α_t)ε, t)||²]
    
    where:
    - ratio: from time-dependent classifier h_omega(x_t, t) = (1-c)/c
    - retention_coeff = ratio^t (ratio raised to power t)
    - reconsolidation_coeff = 1 (fixed)
    
    Enhanced version: Uses up to 3x more source samples per epoch than target samples
    """
    # Calculate the maximum number of source samples to use per epoch
    target_size = target_data.size(0)
    source_size = source_data.size(0)
    max_source_per_epoch = min(3 * target_size, source_size)
    
    print(f"Diff-Tuning Ratio: Using up to {max_source_per_epoch} source samples per epoch (target: {target_size})")
    
    if batch_size > min(max_source_per_epoch, target_size):
        batch_size = min(max_source_per_epoch, target_size)
    
    # Create datasets
    source_dataset = TensorDataset(source_data)
    target_dataset = TensorDataset(target_data)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)
    classifier.eval()  # Time-dependent classifier is assumed pre-trained/frozen

    for epoch in range(n_epochs):
        # Create iterators for both datasets
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Calculate number of batches for target (this determines the epoch length)
        n_target_batches = len(target_loader)
        
        # Calculate how many source batches we can use (up to 3x target batches)
        n_source_batches = min(3 * n_target_batches, len(source_loader))
        
        # Process all target batches
        for batch_idx in range(n_target_batches):
            # Get target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            x0_target = target_batch[0].to(device)      # shape [batch, 2]
            
            # Sample time steps for target
            t_target = torch.rand(x0_target.size(0), device=device)  # uniform in [0,1]
            
            # Forward diffusion for target data
            a_t_target = noise_schedule.marginal_alpha(t_target).unsqueeze(1)
            s_t_target = noise_schedule.marginal_std(t_target).unsqueeze(1)
            eps_target = torch.randn_like(x0_target)
            x_t_target = a_t_target * x0_target + s_t_target * eps_target
            
            # Predict noise for target
            eps_pred_target = model(x_t_target, t_target)
            
            # Compute base loss for target
            base_loss_target = mse_loss(eps_pred_target, eps_target)
            
            # Compute reconsolidation coefficient: fixed at 1
            reconsolidation_coeff = t_target  # shape [batch]
            
            # Compute adaptation loss
            adaptation_loss = torch.mean(reconsolidation_coeff * base_loss_target)
            
            # Process source data (up to 3 batches per target batch)
            source_batches_processed = 0
            max_source_batches_per_target = min(3, n_source_batches // n_target_batches)
            
            retention_losses = []
            while source_batches_processed < max_source_batches_per_target and source_batches_processed < n_source_batches:
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)
                
                x0_source = source_batch[0].to(device)    # shape [batch, 2]
                
                # Sample time steps for source
                t_source = torch.rand(x0_source.size(0), device=device)  # uniform in [0,1]
                
                # Forward diffusion for source data
                a_t_source = noise_schedule.marginal_alpha(t_source).unsqueeze(1)
                s_t_source = noise_schedule.marginal_std(t_source).unsqueeze(1)
                eps_source = torch.randn_like(x0_source)
                x_t_source = a_t_source * x0_source + s_t_source * eps_source
                
                # Predict noise for source
                eps_pred_source = model(x_t_source, t_source)
                
                # Compute base loss for source
                base_loss_source = mse_loss(eps_pred_source, eps_source)
                
                # Get ratio from classifier for source data
                with torch.no_grad():
                    classifier_out = classifier(x0_source)
                    
                    # Clamp classifier output to prevent extreme ratios
                    classifier_out_clamped = torch.clamp(classifier_out, min=1e-8, max=1-1e-8)
                    ratio = (1 - classifier_out_clamped) / classifier_out_clamped  # h_omega(x_t, t) = q_t/p_t
                    
                    # Clamp the final ratio to reasonable range [1e-8, 1e8]
                    ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                
                # Compute retention coefficient: ratio^t
                retention_coeff = ratio * (1 - t_source)  # shape [batch]
                
                # Compute retention loss
                retention_loss = torch.mean(retention_coeff * base_loss_source)
                retention_losses.append(retention_loss)
                
                source_batches_processed += 1
            
            # Average retention loss over all source batches processed
            if retention_losses:
                avg_retention_loss = torch.stack(retention_losses).mean()
            else:
                avg_retention_loss = torch.tensor(0.0, device=device)
            
            # Total loss: L_retention(θ) + L_adaptation(θ)
            total_loss = avg_retention_loss + adaptation_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion Diff-Tuning Ratio] epoch {epoch+1}/{n_epochs}, "
                  f"total_loss={total_loss.item():.4f}, "
                  f"retention_loss={avg_retention_loss.item():.4f}, "
                  f"adaptation_loss={adaptation_loss.item():.4f}")
            
        # Memory cleanup every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.cuda.empty_cache() if device != 'cpu' else None

    model.to('cpu')
    classifier.to('cpu')


def train_diffusion_noise_prediction_diff_tuning_ratio_v3(
    model, source_data, target_data, classifier, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'
):
    """
    Diff-Tuning Ratio V3: Combined Knowledge Retention and Knowledge Reconsolidation
    
    Loss = L_retention(θ) + L_adaptation(θ)
    
    L_retention(θ) = E_{t,ε,x̂₀ˢ~X̂ˢ} [ratio * ||ε - f_θ(√(α_t)x̂₀ˢ + √(1-α_t)ε, t)||²]
    L_adaptation(θ) = E_{t,ε,x₀~X} [1 * ||ε - f_θ(√(α_t)x₀ + √(1-α_t)ε, t)||²]
    
    where:
    - ratio: from time-dependent classifier h_omega(x_t, t) = (1-c)/c
    - retention_coeff = ratio (ratio directly, not raised to power t)
    - reconsolidation_coeff = 1 (fixed)
    
    Enhanced version: Uses up to 3x more source samples per epoch than target samples
    """
    # Calculate the maximum number of source samples to use per epoch
    target_size = target_data.size(0)
    source_size = source_data.size(0)
    max_source_per_epoch = min(3 * target_size, source_size)
    
    print(f"Diff-Tuning Ratio V3: Using up to {max_source_per_epoch} source samples per epoch (target: {target_size})")
    
    if batch_size > min(max_source_per_epoch, target_size):
        batch_size = min(max_source_per_epoch, target_size)
    
    # Create datasets
    source_dataset = TensorDataset(source_data)
    target_dataset = TensorDataset(target_data)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)
    classifier.eval()  # Time-dependent classifier is assumed pre-trained/frozen

    for epoch in range(n_epochs):
        # Create iterators for both datasets
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Calculate number of batches for target (this determines the epoch length)
        n_target_batches = len(target_loader)
        
        # Calculate how many source batches we can use (up to 3x target batches)
        n_source_batches = min(3 * n_target_batches, len(source_loader))
        
        total_loss = 0.0
        
        for batch_idx in range(n_target_batches):
            # Process target batch
            try:
                target_batch = next(target_iter)
                target_x0 = target_batch[0].to(device)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
                target_x0 = target_batch[0].to(device)
            
            batch_size_actual = target_x0.size(0)
            
            # Sample time steps for target data
            t_int = torch.randint(low=1, high=25 + 1, size=(batch_size_actual,), device=device)
            t = t_int.float() / float(25)
            
            # Reconsolidation coefficient: fixed at 1
            reconsolidation_coeff = torch.ones_like(t)
            
            # Add noise to target data
            alpha_t = noise_schedule.marginal_alpha(t).unsqueeze(1)
            sigma_t = noise_schedule.marginal_std(t).unsqueeze(1)
            eps_target = torch.randn_like(target_x0)
            x_t_target = alpha_t * target_x0 + sigma_t * eps_target
            
            # Predict noise for target data
            eps_pred_target = model(x_t_target, t)
            base_loss_target = mse_loss(eps_pred_target, eps_target)
            adaptation_loss = torch.mean(reconsolidation_coeff * base_loss_target)
            
            # Process source batches (up to 3x target batches)
            source_batches_processed = 0
            max_source_batches_per_target = min(3, n_source_batches // n_target_batches)
            
            retention_losses = []
            
            while source_batches_processed < max_source_batches_per_target and source_batches_processed < n_source_batches:
                try:
                    source_batch = next(source_iter)
                    source_x0 = source_batch[0].to(device)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)
                    source_x0 = source_batch[0].to(device)
                
                batch_size_actual_source = source_x0.size(0)
                
                # Sample time steps for source data
                t_int_source = torch.randint(low=1, high=25 + 1, size=(batch_size_actual_source,), device=device)
                t_source = t_int_source.float() / float(25)
                
                # Add noise to source data
                alpha_t_source = noise_schedule.marginal_alpha(t_source).unsqueeze(1)
                sigma_t_source = noise_schedule.marginal_std(t_source).unsqueeze(1)
                eps_source = torch.randn_like(source_x0)
                x_t_source = alpha_t_source * source_x0 + sigma_t_source * eps_source
                
                # Get ratio from classifier
                with torch.no_grad():
                    classifier_out = classifier(source_x0)
                    classifier_out_clamped = torch.clamp(classifier_out, min=1e-8, max=1-1e-8)
                    ratio = (1 - classifier_out_clamped) / classifier_out_clamped
                    ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                
                # Retention coefficient: ratio directly (not raised to power t)
                retention_coeff = ratio
                
                # Predict noise for source data
                eps_pred_source = model(x_t_source, t_source)
                base_loss_source = mse_loss(eps_pred_source, eps_source)
                retention_loss = torch.mean(retention_coeff * base_loss_source)
                retention_losses.append(retention_loss)
                
                source_batches_processed += 1
            
            # Average retention loss from multiple source batches
            if retention_losses:
                avg_retention_loss = torch.stack(retention_losses).mean()
            else:
                avg_retention_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = avg_retention_loss + adaptation_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion Diff-Tuning Ratio V3] epoch {epoch+1}/{n_epochs}, "
                  f"total_loss={total_loss.item():.4f}, "
                  f"retention_loss={avg_retention_loss.item():.4f}, "
                  f"adaptation_loss={adaptation_loss.item():.4f}")
            
        # Memory cleanup every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.cuda.empty_cache() if device != 'cpu' else None

    model.to('cpu')
    classifier.to('cpu')


# -----------------------------------------------------------------------
# 4. Sample from the Noise Predictor and DPM-Solver: https://github.com/LuChengTHU/dpm-solver
# -----------------------------------------------------------------------
class DPM_Solver: # richtig
    """
    A minimal DPM-Solver for unconditional noise-based models.
    We show only a simple single-step solver (DPM-Solver-1) for demonstration.
    For higher-order methods, see the original code.
    """
    def __init__(self, model_fn, noise_schedule, guidance_scale=[0,0], guidance_network=None, 
                 resampling_method=None, classifier=None, resampling_steps=None, method_name=None):
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        self.algorithm_type = "dpmsolver"  # or "dpmsolver++"
        self.eta = 0.9  # extra noise
        self.guidance_network = guidance_network
        self.scale1 = guidance_scale[0]
        self.scale2 = guidance_scale[1]
        self.resampling_method = resampling_method  # 'multinomial' or 'ot'
        self.classifier = classifier
        self.resampling_steps = resampling_steps or []  # List of steps to resample
        self.particle_trajectory = []  # Store particle positions at each step
        self.method_name = method_name or "Unknown Method"  # Method name for animation

    def compute_guidance_noise(self, x, t_, guidance_network, noise_schedule):
        """
        Returns the guidance 'noise' that, when added to the source-model's predicted noise,
        effectively adds ∇_x log h_psi(x,t_) to the total score.

        guidance_noise = - sigma_t * ∇_x log h_psi(x,t_).
        """
        # 1. Require gradient on x
        x = x.detach()
        x.requires_grad_(True)

        # 2. Evaluate h_psi(x, t_) (scalar per sample)
        t_ = t_.unsqueeze(1).expand(x.shape[0], -1)
        h_val = guidance_network(torch.cat([x, t_], dim=1))  # shape [batch,1]

        # 3. sum(log(...)) so that grad(...) w.r.t. x is the sum of ∇ log(h_val_i)
        log_sum = torch.log(h_val + 1e-20).sum()
        grad_log_h = torch.autograd.grad(log_sum, x, create_graph=False)[0]  # shape [batch, d]

        # 4. Convert that gradient to "noise" by multiplying by -sigma_t
        sigma_t = noise_schedule.marginal_std(t_)
        guidance_noise = -sigma_t * grad_log_h

        x.requires_grad_(False)
        return guidance_noise.detach()


    def second_order_update_guidance(self, x, s, t):
        """
        Single-step second-order update from DPM-Solver,
        now with an added guidance gradient.

        r1: A float. The hyperparameter of the second-order solver.
        In the second-order update, we take a step of size r1 from s to s1.
        Then we take a step of s1 to t.
        """
        r1 = 0.5 # In the Pseudo Code they use 0.5 (Algorithm 1 DPM-Solver-2.) on page 6 of the Paper
        scale1 = self.scale1
        scale2 = self.scale2
        ns = self.noise_schedule
        lam_s = ns.marginal_lambda(s)
        lam_t = ns.marginal_lambda(t)
        h = lam_t - lam_s
        lam_s1 = lam_s + r1 * h
        s1 = ns.inverse_lambda(lam_s1)

        model_s_source = self.model(x, s)  # shape [batch, d], source model’s predicted noise
        guidance_s = self.compute_guidance_noise(x, s, self.guidance_network, ns) if self.guidance_network is not None else model_s_source
        model_s_combined = model_s_source + scale1 * guidance_s

        # Step to intermediate s1
        alpha_s1 = torch.exp(ns.marginal_log_mean_coeff(s1))
        sigma_s1 = ns.marginal_std(s1)
        phi_11 = torch.expm1(r1 * h)
        x_s1 = (
                torch.exp(ns.marginal_log_mean_coeff(s1) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_s1 * phi_11 * model_s_combined)

        model_s1_source = self.model(x_s1, s1)
        guidance_s1 = self.compute_guidance_noise(x_s1, s1, self.guidance_network, ns) if self.guidance_network is not None else model_s1_source
        model_s1_combined = model_s1_source + scale2*guidance_s1

        # Final step to t (2nd-order update)
        alpha_t = torch.exp(ns.marginal_log_mean_coeff(t))
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)

        x_t = (
                torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_t * phi_1 * model_s_combined
                - 0.5 * (sigma_t * phi_1) * (model_s1_combined - model_s_combined)
        )

        return x_t

    def first_order_update_guidance(self, x, s, t):
        """
        Single-step first-order update from DPM-Solver,
        with guidance gradient.
        """
        ns = self.noise_schedule
        lam_s = ns.marginal_lambda(s)
        lam_t = ns.marginal_lambda(t)
        h = lam_t - lam_s

        model_s_source = self.model(x, s)  # shape [batch, d], source model’s predicted noise
        guidance_s = self.compute_guidance_noise(x, s, self.guidance_network, ns) if self.guidance_network is not None else model_s_source
        model_s_combined = model_s_source + guidance_s * self.scale1

        # Final step to t (1st-order update)
        alpha_t = torch.exp(ns.marginal_log_mean_coeff(t))
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)

        x_t = (
                torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_t * phi_1 * model_s_combined
        )

        return x_t


    def second_order_sample_guidance(self, x, steps=10, t_start=1.0, t_end=1e-3):
        sample_batch_size = min(1000, x.size(0))
        x = x.to('cpu')
        x_batches = x.split(sample_batch_size)
        x_out_batches = []

        for i, x_batch in enumerate(x_batches):
            device = x_batch.device
            ts = torch.linspace(t_start, t_end, steps+1).to(device)

            # 初始时刻（较大的噪声时间）s=ts[0]，状态为 x_prev
            s = ts[0].unsqueeze(0)
            x_prev = x_batch.clone()
            x_cur = x_batch

            batch_trajectory = [x_cur.clone().detach()]

            for step in range(1, steps+1):
                t = ts[step].unsqueeze(0)

                # 从 s -> t 的更新
                if step == 1:
                    x_cur = self.first_order_update_guidance(x_cur, s, t)
                else:
                    x_cur = self.second_order_update_guidance(x_cur, s, t)

                # ====== Resampling: use phi_t/phi_{t+1} ======
                if self.resampling_method and step in self.resampling_steps and self.classifier is not None:
                    with torch.no_grad():
                        # 估计 phi_t = p_t/q_t at (x_cur, t)
                        if hasattr(self.classifier, 'use_sigmoid') and self.classifier.use_sigmoid:
                            t_exp_t = t.expand(x_cur.shape[0])
                            out_t = self.classifier(x_cur, t_exp_t)   # P(sample from p | x_t, t)
                            t_exp_s = s.expand(x_prev.shape[0])
                            out_s = self.classifier(x_prev, t_exp_s)  # P(sample from p | x_{t+1}, s)
                        else:
                            out_t = self.classifier(x_cur)
                            out_s = self.classifier(x_prev)

                        out_t = torch.clamp(out_t, 1e-4, 1-1e-4)
                        out_s = torch.clamp(out_s, 1e-4, 1-1e-4)

                        # odds = p/q
                        phi_t = out_t / (1.0 - out_t)
                        phi_s = out_s / (1.0 - out_s)

                        # 目标权重：phi_t / phi_{t+1} ；注意我们这里 s = t_{old} = t+1（论文记号）
                        weights = (phi_t / phi_s).squeeze()
                        weights = torch.clamp(weights, 1e-12, 1e12)
                        weights = weights / weights.sum()

                        if self.resampling_method == 'multinomial':
                            N = x_cur.shape[0]
                            w_np = weights.detach().cpu().numpy()
                            w_np = np.maximum(w_np, 1e-12)
                            w_np = w_np / w_np.sum()
                            idx = np.random.choice(np.arange(N), size=N, p=w_np)
                            x_cur = x_cur[idx]
                        elif self.resampling_method == 'ot':
                            x_cur = ot_resampling(x_cur, weights)

                # 记录轨迹
                batch_trajectory.append(x_cur.clone().detach())

                # 下一步前把当前作为“上一时刻”
                x_prev = x_cur.clone().detach()
                s = t

            x_out_batches.append(x_cur)

            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache() if device != 'cpu' else None

            if i == 0:
                method_name = self.method_name
                animation_path = f"{results_dir}/{method_name.lower().replace(' ', '_').replace('-', '_')}_animation.gif"
                create_resampling_animation(
                    batch_trajectory,
                    method_name,
                    self.resampling_steps if self.resampling_method else [],
                    animation_path
                )

        return torch.cat(x_out_batches, dim=0)


# -----------------------------------------------------------------------
# # 5. Train Source Model (Always Required)
# # -----------------------------------------------------------------------
# print("Training source model (required for all methods)...")
# noise_predictor_target_finetune = NoisePredictor()
# train_diffusion_noise_prediction(noise_predictor_target_finetune, source_data, source_epochs, batch_size, learning_rate, device)
# dpm_solver = DPM_Solver(noise_predictor_target_finetune, noise_schedule)
# x_init = torch.randn(5000, 2)
# x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# plot_true_and_diffused(true_data=source_data_plot, diffused_data=x_out, true_labels=source_labels_plot, diffused_labels=None
#                        , title_true_data="True Source Distribution (Trained on)", title_diffused_data="Pre-trained DM Sampled Distribution",
#                        save_name=f"results/enhance_net/pretrained_source_n{args.n}")

# # Calculate likelihood for pretrained model (commented out for cosine modulation)
# # pdf_vals_pretrained = gaussian_pdf(x_out, mu_S, sigma2_source)
# # avg_likelihood_pretrained = pdf_vals_pretrained.mean().item()
# # print(f"Pre-trained model average likelihood: {avg_likelihood_pretrained:.6f}")
# avg_likelihood_pretrained = 0.0
# print("Likelihood calculations commented out for cosine modulation example")

# # Set default values for baseline variables
# avg_likelihood_pretrained = 0.0
# avg_likelihood_vanilla = 0.0
# avg_likelihood_finetuned = 0.0

if show_baseline:
    print("\n" + "="*80)
    print("RUNNING BASELINE METHODS")
    print("="*80)
    
    # -----------------------------------------------------------------------
    # 6. Vanilla Plot Train the Noise Predictor on the Target Domain
    # -----------------------------------------------------------------------
    noise_predictor_target = NoisePredictor()
    train_diffusion_noise_prediction(noise_predictor_target,target_data_, vanilla_diffusion_epochs, target_batch_size, learning_rate,device)
    dpm_solver = DPM_Solver(noise_predictor_target, noise_schedule, method_name="Vanilla Diffusion")
    x_init = torch.randn(5000, 2)
    x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
    plot_true_and_diffused(target_data_plot, x_out, target_labels_plot, None,
                           title_true_data="True Target Distribution (Trained on)", title_diffused_data="Vanilla DM Sampled Distribution",
                           save_name=f"{results_dir}/vanilla_diffusion_n{args.n}")
    
    # Calculate likelihood for vanilla diffusion using Gaussian distributions
    target_likelihood_vanilla = compute_target_likelihood(x_out, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device)
    source_likelihood_vanilla = compute_source_likelihood(x_out, mu_source, sigma2_source, device)
    avg_likelihood_vanilla = target_likelihood_vanilla.mean().item()
    print(f"Vanilla diffusion - Target likelihood: {avg_likelihood_vanilla:.6f}")
    print(f"Vanilla diffusion - Source likelihood: {source_likelihood_vanilla.mean().item():.6f}")
    print(f"Vanilla diffusion - Likelihood ratio (Target/Source): {(target_likelihood_vanilla.mean() / source_likelihood_vanilla.mean()).item():.6f}")

    # -----------------------------------------------------------------------
    # 7. Finetune the Noise Predictor on the Target Domain
    # -----------------------------------------------------------------------
    # First train a source model, then finetune it on target domain
    print("Training source model for finetuning...")
    noise_predictor_source_for_finetune = NoisePredictor()
    train_diffusion_noise_prediction(noise_predictor_source_for_finetune, source_data, source_epochs, batch_size, learning_rate, device)
    dpm_solver = DPM_Solver(noise_predictor_source_for_finetune, noise_schedule, method_name="Source Model")
    x_init = torch.randn(5000, 2)
    x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
    plot_true_and_diffused(true_data=source_data_plot, diffused_data=x_out, true_labels=source_labels_plot, diffused_labels=None,
                           title_true_data="True Source Distribution (Trained on)", title_diffused_data="Pre-trained DM Sampled Distribution",
                           save_name=f"{results_dir}/pretrained_source_n{args.n}")

    
    # Now finetune the source model on target domain
    print("Finetuning source model on target domain...")
    noise_predictor_target_finetune = noise_predictor_source_for_finetune  # Start from source model
    train_diffusion_noise_prediction(noise_predictor_target_finetune, target_data_, finetune_diffusion_epochs, batch_size, learning_rate, device)
    
    dpm_solver = DPM_Solver(noise_predictor_target_finetune, noise_schedule, method_name="Finetuned Target")
    x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
    plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out, true_labels=target_labels_plot, diffused_labels=None,
                           title_true_data="True Target Distribution (Fine-Tuned on)", title_diffused_data="Finetuned DM Sampled Distribution",
                           save_name=f"{results_dir}/finetuned_target_n{args.n}")
    
    # Calculate likelihood for finetuned model using Gaussian distributions
    target_likelihood_finetuned = compute_target_likelihood(x_out, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device)
    source_likelihood_finetuned = compute_source_likelihood(x_out, mu_source, sigma2_source, device)
    avg_likelihood_finetuned = target_likelihood_finetuned.mean().item()
    print(f"Finetuned model - Target likelihood: {avg_likelihood_finetuned:.6f}")
    print(f"Finetuned model - Source likelihood: {source_likelihood_finetuned.mean().item():.6f}")
    print(f"Finetuned model - Likelihood ratio (Target/Source): {(target_likelihood_finetuned.mean() / source_likelihood_finetuned.mean()).item():.6f}")

gc.collect()
torch.cuda.empty_cache()




# -----------------------------------------------------------------------
# 7. Train the Domain Classifier and the Guidance Network
#   - Pseudo Code Algorithm 1 and Algorithm 3
# -----------------------------------------------------------------------
def train_domain_classifier(model, source_data, target_data, n_epochs=5, batch_size=512, lr=1e-8, device='cpu'):
    """
    Minimal training loop for domain classification on 2D data.
    """
    source_labels = torch.ones(source_data.size(0))
    target_labels = torch.zeros(target_data.size(0))

    dataset = TensorDataset(torch.cat([source_data, target_data]), torch.cat([source_labels, target_labels]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()

    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(x)
            loss = bce_loss(y_pred, y.unsqueeze(1).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Classifier] epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")
    model.to('cpu')
    return model


def train_guidance_network(
        guidance_network,
        classifier,
        noise_schedule,
        source_data,
        T=25,
        n_epochs=20,
        batch_size=512,
        lr=1e-4,
        device='cpu'
):
    """
    Trains 'guidance_network' using the objective:
        L(psi) = E_{x0, t, x_t} [|| h_psi(x_t, t) - c_omega(x0) ||^2_2],
    where x_t = alpha_t * x0 + sigma_t * eps.
    """
    print(f"Training guidance network with ratio clamping [1e-8, 1e8]")
    print(f"Using classifier: {type(classifier).__name__}")
    dataset = TensorDataset(source_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    guidance_network.to(device)
    classifier.to(device)
    classifier.eval()  # Classifier is assumed pre-trained/frozen

    optimizer = optim.Adam(guidance_network.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 2. Training loop
    count_update = 0
    for epoch in range(n_epochs):
        for batch in loader:
            x0 = batch[0].to(device)  # shape [batch, d]

            # Sample discrete t from {1,...,T} uniformly
            t_int = torch.randint(low=1, high=T + 1, size=(x0.size(0),), device=device)
            t_ = t_int.float() / float(T)

            # 3. Forward diffuse: x_t = alpha_t * x0 + sigma_t * eps
            alpha_t = noise_schedule.marginal_alpha(t_).unsqueeze(1)  # shape [batch, 1]
            sigma_t = noise_schedule.marginal_std(t_).unsqueeze(1)  # shape [batch, 1]
            eps = torch.randn_like(x0)
            x_t = alpha_t * x0 + sigma_t * eps

            # 4. Compute guidance_network outputs and domain classifier target
            with torch.no_grad():
                # c_omega(x0) is the classifier output for the original data
                classifier_out = classifier(x0)  # shape [batch, 1]

            guidance_out = guidance_network(torch.cat([x_t, t_.unsqueeze(1)], dim=1))

            # 5. Compute custom guidance loss (Algorithm 2)

            # Clamp classifier output to prevent extreme ratios
            classifier_out_clamped = torch.clamp(classifier_out, min=1e-8, max=1-1e-8)
            target = (1 - classifier_out_clamped) / classifier_out_clamped  # Algorithm 4 in the Paper
            
            # Clamp the final ratio to reasonable range [1e-8, 1e8]
            target = torch.clamp(target, min=1e-8, max=1e8)
            
            # target = classifier_out # Algorithm 3 in the Paper
            # This is the original code, but I think it is wrong, the guidance loss should as in Algorithm 4
            loss = torch.mean(torch.sum((guidance_out - target) ** 2, dim=1))

            # 6. Update guidance_network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count_update += 1

        # (Optional) Print some progress
        if (epoch + 1) % 10 == 0:
            print(f"[Guidance] epoch {epoch + 1}/{n_epochs}, loss={loss.item():.4f}")

    # Return trained model weights
    print(f"Guidance network training completed. Total updates: {count_update}")
    guidance_network.to('cpu')
    return guidance_network


# -----------------------------------------------------------------------
# 8. Train the Domain Classifier and the Guidance Network with Regularization
#   - Pseudo Code Algorithm 2 and Algorithm 4
# -----------------------------------------------------------------------
def train_time_dependent_classifier(
        classifier_time_dependent,
        source_data,
        target_data,
        noise_schedule,
        T=25,
        n_epochs=20,
        batch_size=512,
        lr=1e-4,
        device='cpu',
        samples_per_x0=10,  # 每个x0生成多少个时间步样本
        use_multiple_t=True,  # 是否使用多时间步训练
        enable_balancing=True):  # 是否启用样本平衡
    # 可选的样本平衡
    if enable_balancing:
        # 确保样本比例不超过1:10 (source:target)
        source_size = source_data.size(0)
        target_size = target_data.size(0)
        
        # 计算平衡后的样本数量
        if source_size > target_size * 1:
            # source太多，限制source数量
            source_balanced = source_data[:target_size * 1]
            target_balanced = target_data
            print(f"Balancing: source {source_size} -> {source_balanced.size(0)}, target {target_size}")
        elif target_size > source_size * 1:
            # target太多，限制target数量
            source_balanced = source_data
            target_balanced = target_data[:source_size * 1]
            print(f"Balancing: source {source_size}, target {target_size} -> {target_balanced.size(0)}")
        else:
            # 比例合理，使用全部数据
            source_balanced = source_data
            target_balanced = target_data
            print(f"No balancing needed: source {source_size}, target {target_size}")
    else:
        # 不使用平衡，直接使用全部数据
        source_balanced = source_data
        target_balanced = target_data
        print(f"No balancing (disabled): source {source_data.size(0)}, target {target_data.size(0)}")
    
    # Combine balanced source and target data
    dataset = TensorDataset(
        torch.cat([source_balanced, target_balanced]),
        torch.cat([torch.ones(source_balanced.size(0)), torch.zeros(target_balanced.size(0))])
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    classifier_time_dependent.to(device)
    optimizer = optim.Adam(classifier_time_dependent.parameters(), lr=lr)
    bce_loss = nn.BCELoss()
    if not use_multiple_t:
        n_epochs = n_epochs * 2

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for x0, labels in loader:
            x0, labels = x0.to(device), labels.to(device)
            batch_size_actual = x0.size(0)
            
            if use_multiple_t:
                # 为每个x0生成多个时间步样本
                # 扩展数据：每个x0生成samples_per_x0个样本
                x0_expanded = x0.repeat_interleave(samples_per_x0, dim=0)  # [batch_size * samples_per_x0, 2]
                labels_expanded = labels.repeat_interleave(samples_per_x0, dim=0)  # [batch_size * samples_per_x0]
                
                # 为每个样本采样不同的时间步
                t_int = torch.randint(low=1, high=T + 1, size=(x0_expanded.size(0),), device=device)
                t = t_int.float() / float(T)
                
                # 确保每个x0的最后一个副本是原始x0，不受噪声干扰
                for i in range(batch_size_actual):
                    # 找到每个x0的最后一个副本的索引
                    last_idx = (i + 1) * samples_per_x0 - 1
                    t_int[last_idx] = 0  # 设置t=0，表示无噪声
                    t[last_idx] = 0.0    # 对应的时间步为0
                
                alpha_t = noise_schedule.marginal_alpha(t).unsqueeze(1)  # shape [batch, 1]
                sigma_t = noise_schedule.marginal_std(t).unsqueeze(1)  # shape [batch, 1]
                eps = torch.randn_like(x0_expanded)
                x_t = alpha_t * x0_expanded + sigma_t * eps
                
                # 确保t=0的样本是干净的
                zero_noise_mask = (t == 0.0).unsqueeze(1)
                x_t = torch.where(zero_noise_mask, x0_expanded, x_t)
                
                # 前向传播
                predictions = classifier_time_dependent(x_t, t)
                loss = bce_loss(predictions, labels_expanded.unsqueeze(1))
                
                total_samples = batch_size_actual * samples_per_x0
            else:
                # 原始方法：每个x0只生成1个时间步样本
                t_int = torch.randint(low=1, high=T + 1, size=(batch_size_actual,), device=device)
                t = t_int.float() / float(T)
                
                alpha_t = noise_schedule.marginal_alpha(t).unsqueeze(1)  # shape [batch, 1]
                sigma_t = noise_schedule.marginal_std(t).unsqueeze(1)  # shape [batch, 1]
                eps = torch.randn_like(x0)
                x_t = alpha_t * x0 + sigma_t * eps

                # 前向传播
                predictions = classifier_time_dependent(x_t, t)
                loss = bce_loss(predictions, labels.unsqueeze(1))
                
                total_samples = batch_size_actual
            

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        if (epoch + 1) % 10 == 0:
            print(f"[Time Dependent Classifier] epoch {epoch + 1}/{n_epochs}, loss={avg_loss:.4f}")
            
            # 添加调试信息
            if epoch == 0:  # 只在第一个epoch打印详细信息
                print(f"  - 数据形状: x0={x0.shape}, labels={labels.shape}")
                if use_multiple_t:
                    print(f"  - 扩展后: x0_expanded={x0_expanded.shape}, labels_expanded={labels_expanded.shape}")
                    print(f"  - 时间步范围: t_min={t.min():.4f}, t_max={t.max():.4f}")
                    print(f"  - 预测值范围: pred_min={predictions.min():.4f}, pred_max={predictions.max():.4f}")
                    print(f"  - 标签分布: source={labels_expanded.sum().item()}, target={len(labels_expanded)-labels_expanded.sum().item()}")
                else:
                    print(f"  - 时间步范围: t_min={t.min():.4f}, t_max={t.max():.4f}")
                    print(f"  - 预测值范围: pred_min={predictions.min():.4f}, pred_max={predictions.max():.4f}")
                    print(f"  - 标签分布: source={labels.sum().item()}, target={len(labels)-labels.sum().item()}")

    classifier_time_dependent.to('cpu')
    return classifier_time_dependent



# Reuse the source model from finetuning for all guidance and resampling methods
print("Reusing source model from finetuning for guidance and resampling methods...")
noise_predictor_source = noise_predictor_source_for_finetune  # Reuse the already trained source model
print("Source model ready for guidance and resampling methods")

# -----------------------------------------------------------------------
# 8. Two Guidance Network Training Schemes
# -----------------------------------------------------------------------

# ====================================================================
# SCHEME 1: CLASSIFIER-BASED GUIDANCE (ORIGINAL METHOD)
# ====================================================================
print("\n" + "="*80)
print("SCHEME 1: CLASSIFIER-BASED GUIDANCE (ORIGINAL METHOD)")
print("="*80)
print("Description: Uses all source data to train classifier, then trains guidance network")
print("Classifier: Trained on all source data (unbalanced)")
print("Guidance: Standard guidance network without regularization")
print("="*80)

classifier = Classifier()
classifier = train_domain_classifier(classifier, source_data, target_data_, guidance_epochs, batch_size, learning_rate, device)
evaluate_domain_classifier(model=classifier, source_data=source_data, target_data=target_data_, device=device)

# if use_regularization :
# classifier_time_dependent = Classifier(input_dim=d+1)
# classifier_time_dependent = train_time_dependent_classifier(classifier_time_dependent, source_data, target_data_, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)
    # guidance_network_classifier = GuidanceNetwork()
    # guidance_network_classifier = train_guidance_network_with_regularization(guidance_network_classifier, classifier, classifier_time_dependent, noise_predictor_tgdp, noise_schedule, source_data, target_data_,
                                                                #   eta1and2=eta1and2, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)
# else:
guidance_network_classifier = GuidanceNetwork()
guidance_network_classifier = train_guidance_network(guidance_network_classifier, classifier, noise_schedule, source_data,
                                            T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size,
                                            lr=learning_rate, device=device)

# Test classifier-based guidance
dpm_solver_classifier = DPM_Solver(model_fn=noise_predictor_source, guidance_scale=guidance_scale, noise_schedule=noise_schedule, guidance_network=guidance_network_classifier, method_name="Classifier-based Guidance")
x_init = torch.randn(5000, 2)
x_out_classifier = dpm_solver_classifier.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)

# Determine the scheme name for classifier-based guidance
if use_regularization:
    scheme_name_classifier = f"guidance_classifier_eta{eta1and2[0]}_{eta1and2[1]}_n{args.n}"
else:
    scheme_name_classifier = f"guidance_classifier_no_reg_n{args.n}"

plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out_classifier, true_labels=target_labels_plot, diffused_labels=None
                       , title_true_data="Target Distribution", title_diffused_data=f"Classifier-based Guidance",
                       save_name=f"{results_dir}/{scheme_name_classifier}")

# Calculate likelihood for classifier-based guidance using Gaussian distributions
target_likelihood_guidance_classifier = compute_target_likelihood(x_out_classifier, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device)
source_likelihood_guidance_classifier = compute_source_likelihood(x_out_classifier, mu_source, sigma2_source, device)
avg_likelihood_guidance_classifier = target_likelihood_guidance_classifier.mean().item()
print(f"Classifier-based guidance - Target likelihood: {avg_likelihood_guidance_classifier:.6f}")
print(f"Classifier-based guidance - Source likelihood: {source_likelihood_guidance_classifier.mean().item():.6f}")
print(f"Classifier-based guidance - Likelihood ratio (Target/Source): {(target_likelihood_guidance_classifier.mean() / source_likelihood_guidance_classifier.mean()).item():.6f}")

# ====================================================================
# SCHEME 2: BALANCED TRAINING GUIDANCE
# ====================================================================
print("\n" + "="*80)
print("SCHEME 2: BALANCED TRAINING GUIDANCE")
print("="*80)
print("Description: Uses balanced source/target data to train classifier, then trains guidance network")
print("Classifier: Trained on balanced source/target data (1:1 ratio)")
print("Guidance: Standard guidance network without regularization")
print("="*80)

# Create balanced dataset: sample same number of source data as target data
if source_data.size(0) > target_data_.size(0):
    # Randomly sample source data to match target data size
    indices = torch.randperm(source_data.size(0))[:target_data_.size(0)]
    source_data_balanced = source_data[indices]
    source_labels_balanced = source_labels[indices]
    print(f"Balanced source data: {source_data_balanced.size(0)} samples (originally {source_data.size(0)})")
else:
    source_data_balanced = source_data
    source_labels_balanced = source_labels
    print(f"Using all source data: {source_data_balanced.size(0)} samples")

# Train classifier on balanced data
# Ensure batch_size is not larger than available data
total_samples = source_data_balanced.size(0) + target_data_.size(0)
print(f"Balanced data: {source_data_balanced.size(0)} source + {target_data_.size(0)} target = {total_samples} total")
classifier_balanced = Classifier()
classifier_balanced = train_domain_classifier(classifier_balanced, source_data_balanced, target_data_, guidance_epochs, 256, learning_rate, device)
evaluate_domain_classifier(model=classifier_balanced, source_data=source_data_balanced, target_data=target_data_, device=device)

# Time-dependent classifiers will be trained before their respective schemes

# if use_regularization :
# guidance_network = GuidanceNetwork()
# guidance_network = train_guidance_network_with_regularization(guidance_network, classifier_balanced, classifier_time_dependent_balanced, noise_predictor_tgdp, noise_schedule, source_data, target_data_,
#                                                                 eta1and2=eta1and2, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)
# # else:
guidance_network = GuidanceNetwork()
guidance_network = train_guidance_network(guidance_network, classifier_balanced, noise_schedule, source_data,
                                            T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size,
                                            lr=learning_rate, device=device)

# Test balanced guidance
dpm_solver_balanced = DPM_Solver(model_fn=noise_predictor_source, guidance_scale=guidance_scale, noise_schedule=noise_schedule, guidance_network=guidance_network, method_name="Balanced Training Guidance")
x_out_balanced = dpm_solver_balanced.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)

# Determine the scheme name for balanced guidance
if use_regularization:
    scheme_name_balanced = f"guidance_balanced_eta{eta1and2[0]}_{eta1and2[1]}_n{args.n}"
else:
    scheme_name_balanced = f"guidance_balanced_no_reg_n{args.n}"

plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out_balanced, true_labels=target_labels_plot, diffused_labels=None
                       , title_true_data="Target Distribution", title_diffused_data=f"Balanced Training Guidance",
                       save_name=f"{results_dir}/{scheme_name_balanced}")

# Calculate likelihood for balanced training guidance using Gaussian distributions
target_likelihood_guidance_balanced = compute_target_likelihood(x_out_balanced, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device)
source_likelihood_guidance_balanced = compute_source_likelihood(x_out_balanced, mu_source, sigma2_source, device)
avg_likelihood_guidance_balanced = target_likelihood_guidance_balanced.mean().item()
print(f"Balanced training guidance - Target likelihood: {avg_likelihood_guidance_balanced:.6f}")
print(f"Balanced training guidance - Source likelihood: {source_likelihood_guidance_balanced.mean().item():.6f}")
print(f"Balanced training guidance - Likelihood ratio (Target/Source): {(target_likelihood_guidance_balanced.mean() / source_likelihood_guidance_balanced.mean()).item():.6f}")

# ====================================================================
# TRAIN BALANCED TIME-DEPENDENT CLASSIFIER FOR RESAMPLING
# ====================================================================
print("\n" + "="*80)
print("TRAINING BALANCED TIME-DEPENDENT CLASSIFIER FOR RESAMPLING")
print("="*80)
print("Description: Trains time-dependent classifier for resampling methods")
print("Input: (x_t, t) - spatial coordinates and diffusion time")
print("Output: Domain probability for density ratio calculation")
print("="*80)

# Train balanced time-dependent classifier for resampling
classifier_time_dependent_balanced = TimeDependentClassifier(input_dim=d, use_sigmoid=True)
classifier_time_dependent_balanced = train_time_dependent_classifier(
    classifier_time_dependent_balanced, 
    source_data_balanced, 
    target_data_, 
    noise_schedule, 
    T=diffusion_steps, 
    n_epochs=guidance_epochs, 
    batch_size=256,  # Fixed batch size for time-dependent classifier
    lr=learning_rate, 
    device=device, 
    samples_per_x0=10, 
    use_multiple_t=True, 
    enable_balancing=True
)
print("Balanced time-dependent classifier training completed for resampling")

# ====================================================================
# SCHEME 3: MULTINOMIAL RESAMPLING WITH CLASSIFIER
# ====================================================================
print("\n" + "="*80)
print("SCHEME 3: MULTINOMIAL RESAMPLING WITH CLASSIFIER")
print("="*80)
print("Description: Uses time-dependent classifier-based density ratio for multinomial resampling during sampling")
print("Classifier: Balanced time-dependent classifier trained on balanced source/target data")
print("Resampling: Multinomial resampling at specified steps using q/p ratio from time-dependent classifier")
print("="*80)

# Test multinomial resampling with balanced time-dependent classifier
dpm_solver_multinomial = DPM_Solver(
    model_fn=noise_predictor_source, 
    noise_schedule=noise_schedule, 
    guidance_scale=[0, 0],  # Not used in resampling methods
    guidance_network=None,  # No guidance network for resampling
    resampling_method='multinomial',
    classifier=classifier_time_dependent_balanced,  # Use time-dependent classifier
    resampling_steps=resampling_steps,  # Use global resampling steps
    method_name="Multinomial Resampling"
)
x_init = torch.randn(5000, 2)
x_out_multinomial = dpm_solver_multinomial.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out_multinomial, true_labels=target_labels_plot, diffused_labels=None,
                       title_true_data="True Target Distribution", title_diffused_data="Multinomial Resampling",
                       save_name=f"{results_dir}/multinomial_resampling_n{args.n}")
print(f"Multinomial resampling plot saved to: {results_dir}/multinomial_resampling_n{args.n}.png")

# Calculate likelihood for multinomial resampling using Gaussian distributions
target_likelihood_multinomial = compute_target_likelihood(x_out_multinomial, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device)
source_likelihood_multinomial = compute_source_likelihood(x_out_multinomial, mu_source, sigma2_source, device)
avg_likelihood_multinomial = target_likelihood_multinomial.mean().item()
print(f"Multinomial resampling - Target likelihood: {avg_likelihood_multinomial:.6f}")
print(f"Multinomial resampling - Source likelihood: {source_likelihood_multinomial.mean().item():.6f}")
print(f"Multinomial resampling - Likelihood ratio (Target/Source): {(target_likelihood_multinomial.mean() / source_likelihood_multinomial.mean()).item():.6f}")

# ====================================================================
# SCHEME 4: OT RESAMPLING WITH CLASSIFIER
# ====================================================================
print("\n" + "="*80)
print("SCHEME 4: OT RESAMPLING WITH CLASSIFIER")
print("="*80)
print("Description: Uses time-dependent classifier-based density ratio for optimal transport resampling during sampling")
print("Classifier: Balanced time-dependent classifier trained on balanced source/target data")
print("Resampling: Sinkhorn-OT resampling at specified steps using q/p ratio from time-dependent classifier")
print("="*80)

# Test OT resampling with balanced time-dependent classifier
dpm_solver_ot = DPM_Solver(
    model_fn=noise_predictor_source, 
    noise_schedule=noise_schedule, 
    guidance_scale=[0, 0],  # Not used in resampling methods
    guidance_network=None,  # No guidance network for resampling
    resampling_method='ot',
    classifier=classifier_time_dependent_balanced,  # Use time-dependent classifier
    resampling_steps=resampling_steps,  # Use global resampling steps
    method_name="OT Resampling"
)
x_init = torch.randn(5000, 2)
x_out_ot = dpm_solver_ot.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
print("x_out:")
print(x_out_ot)
plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out_ot, true_labels=target_labels_plot, diffused_labels=None,
                       title_true_data="True Target Distribution", title_diffused_data="OT Resampling",
                       save_name=f"{results_dir}/ot_resampling_n{args.n}")
print(f"OT resampling plot saved to: {results_dir}/ot_resampling_n{args.n}.png")

# Calculate likelihood for OT resampling using Gaussian distributions
target_likelihood_ot = compute_target_likelihood(x_out_ot, mu_main, mu_side, sigma2_main, sigma2_side, w_main, device)
source_likelihood_ot = compute_source_likelihood(x_out_ot, mu_source, sigma2_source, device)
avg_likelihood_ot = target_likelihood_ot.mean().item()
print(f"OT resampling - Target likelihood: {avg_likelihood_ot:.6f}")
print(f"OT resampling - Source likelihood: {source_likelihood_ot.mean().item():.6f}")
print(f"OT resampling - Likelihood ratio (Target/Source): {(target_likelihood_ot.mean() / source_likelihood_ot.mean()).item():.6f}")

# ====================================================================
# SCHEME 5: IMPORTANCE SAMPLING (TIME-DEPENDENT CLASSIFIER)
# ====================================================================
print("\n" + "="*80)
print("SCHEME 3: IMPORTANCE SAMPLING (TIME-DEPENDENT CLASSIFIER)")
print("="*80)
print("Description: Uses time-dependent classifier for importance sampling in noise prediction")
print("Classifier: Time-dependent classifier trained on all source data")
print("Training: Importance sampling with ratio weighting: h_omega(x_t, t) * loss")
print("="*80)

# COMMENTED OUT: Train time-dependent classifier with balanced data (no longer needed)
# print(f"Training time-dependent classifier with balanced source/target data")
# classifier_time_dependent = TimeDependentClassifier(input_dim=d, use_sigmoid=True)
# classifier_time_dependent = train_time_dependent_classifier(classifier_time_dependent, source_data, target_data_, noise_schedule, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device, samples_per_x0=10, use_multiple_t=True, enable_balancing=True)

# COMMENTED OUT: Importance sampling results generation
# # Train noise predictor with importance sampling using time-dependent classifier
# noise_predictor_is_unbalanced = NoisePredictor()
# train_diffusion_noise_prediction_importance_sampling(
#     noise_predictor_is_unbalanced, source_data, classifier_time_dependent, 
#     n_epochs=source_epochs, batch_size=batch_size, lr=learning_rate, device=device
# )

# # Sample from the importance sampling trained model (no guidance needed)
# dpm_solver_is_unbalanced = DPM_Solver(noise_predictor_is_unbalanced, noise_schedule)
# x_init = torch.randn(2000, 2)  # Further reduce sample size to avoid memory issues
# print(f"begin to sample")
# x_out = dpm_solver_is_unbalanced.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# print(f"begin to plot")
# plot_true_and_diffused(true_data=target_data_, diffused_data=x_out, true_labels=target_labels_all_n[n_target_index], diffused_labels=None,
#                        save_name=f"results/IS/importance_sampling_unbalanced_n{args.n}")
# print(f"end to plot")
# print(f"Importance sampling (unbalanced) plot saved to: results/IS/importance_sampling_unbalanced_n{args.n}.png")

# # Calculate likelihood for importance sampling (unbalanced)
# pdf_vals_is_unbalanced = mixture_pdf_bimodal(x_out, mu_T_main, mu_T_side, sigma2_main, sigma2_side, w_main)
# avg_likelihood_is_unbalanced = pdf_vals_is_unbalanced.mean().item()
# print(f"Importance sampling (time-dependent) average likelihood: {avg_likelihood_is_unbalanced:.6f}")

# Set default values for commented out methods
avg_likelihood_is_unbalanced = 0.0

# ====================================================================
# SCHEME 4: IMPORTANCE SAMPLING (BALANCED TIME-DEPENDENT CLASSIFIER)
# ====================================================================
print("\n" + "="*80)
print("SCHEME 4: IMPORTANCE SAMPLING (BALANCED TIME-DEPENDENT CLASSIFIER)")
print("="*80)
print("Description: Uses balanced time-dependent classifier for importance sampling")
print("Classifier: Time-dependent classifier trained on balanced source/target data")
print("Training: Importance sampling with ratio weighting: h_omega(x_t, t) * loss")
print("="*80)

# COMMENTED OUT: Balanced importance sampling results generation
# # Train time-dependent classifier for balanced data (needed for Scheme 4)
# print(f"Training time-dependent classifier with balanced data")
# classifier_time_dependent_balanced = TimeDependentClassifier(input_dim=d, use_sigmoid=True)
# classifier_time_dependent_balanced = train_time_dependent_classifier(classifier_time_dependent_balanced, source_data_balanced, target_data_, noise_schedule, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size_classifier_2, lr=learning_rate, device=device, samples_per_x0=3, use_multiple_t=True, enable_balancing=True)

# # Train noise predictor with importance sampling using balanced time-dependent classifier
# noise_predictor_is_balanced = NoisePredictor()
# train_diffusion_noise_prediction_importance_sampling(
#     noise_predictor_is_balanced, source_data, classifier_time_dependent_balanced, 
#     n_epochs=source_epochs, batch_size=batch_size, lr=learning_rate, device=device
# )

# # Sample from the importance sampling trained model (no guidance needed)
# dpm_solver_is_balanced = DPM_Solver(noise_predictor_is_balanced, noise_schedule)
# x_out = dpm_solver_is_balanced.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# plot_true_and_diffused(true_data=target_data_, diffused_data=x_out, true_labels=target_labels_all_n[n_target_index], diffused_labels=None,
#                        save_name=f"results/IS/importance_sampling_balanced_n{args.n}")
# print(f"Importance sampling (balanced) plot saved to: results/IS/importance_sampling_balanced_n{args.n}.png")

# # Calculate likelihood for importance sampling (balanced)
# pdf_vals_is_balanced = mixture_pdf_bimodal(x_out, mu_T_main, mu_T_side, sigma2_main, sigma2_side, w_main)
# avg_likelihood_is_balanced = pdf_vals_is_balanced.mean().item()
# print(f"Importance sampling (balanced time-dep) average likelihood: {avg_likelihood_is_balanced:.6f}")

# Set default values for commented out methods
avg_likelihood_is_balanced = 0.0

# ====================================================================
# SCHEME 5: IMPORTANCE SAMPLING WITH T-POWER RATIO
# ====================================================================
print("\n" + "="*80)
print("SCHEME 5: IMPORTANCE SAMPLING WITH T-POWER RATIO")
print("="*80)
print("Description: Uses time-dependent classifier with ratio raised to power t")
print("Classifier: Time-dependent classifier trained on all source data")
print("Training: Importance sampling with t-power ratio weighting: h_omega(x_t, t)^t * loss")
print("="*80)

# COMMENTED OUT: T-power importance sampling results generation
# # Train noise predictor with t-power importance sampling using time-dependent classifier
# noise_predictor_is_t_power = NoisePredictor()
# train_diffusion_noise_prediction_importance_sampling_t_power(
#     noise_predictor_is_t_power, source_data, classifier_time_dependent, 
#     n_epochs=source_epochs, batch_size=batch_size, lr=learning_rate, device=device
# )

# # Sample from the t-power importance sampling trained model (no guidance needed)
# dpm_solver_is_t_power = DPM_Solver(noise_predictor_is_t_power, noise_schedule)
# x_out = dpm_solver_is_t_power.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# plot_true_and_diffused(true_data=target_data_, diffused_data=x_out, true_labels=target_labels_all_n[n_target_index], diffused_labels=None,
#                        save_name=f"results/IS/importance_sampling_t_power_n{args.n}")
# print(f"Importance sampling (t-power) plot saved to: results/IS/importance_sampling_t_power_n{args.n}.png")

# # Calculate likelihood for importance sampling (t-power)
# pdf_vals_is_t_power = mixture_pdf_bimodal(x_out, mu_T_main, mu_T_side, sigma2_main, sigma2_side, w_main)
# avg_likelihood_is_t_power = pdf_vals_is_t_power.mean().item()
# print(f"Importance sampling (t-power) average likelihood: {avg_likelihood_is_t_power:.6f}")

# Set default values for commented out methods
avg_likelihood_is_t_power = 0.0

# ====================================================================
# ====================================================================
# SCHEME 6: DIFF-TUNING (KNOWLEDGE RETENTION + RECONSOLIDATION) - COMMENTED OUT
# ====================================================================
# print("\n" + "="*80)
# print("SCHEME 6: DIFF-TUNING (KNOWLEDGE RETENTION + RECONSOLIDATION)")
# print("="*80)
# print("Description: Combined Knowledge Retention and Knowledge Reconsolidation")
# print("Training: L_retention(θ) + L_adaptation(θ)")
# print("Retention: ξ(t) * loss on source data (ξ(t) decreasing)")
# print("Adaptation: ψ(t) * loss on target data (ψ(t) increasing)")
# print("="*80)

# # Train noise predictor with Diff-Tuning using both source and target data
# noise_predictor_diff_tuning = NoisePredictor()
# train_diffusion_noise_prediction_diff_tuning(
#     noise_predictor_diff_tuning, source_data, target_data_, 
#     n_epochs=source_epochs, batch_size=batch_size, lr=learning_rate, device=device
# )

# # Sample from the Diff-Tuning trained model (no guidance needed)
# dpm_solver_diff_tuning = DPM_Solver(noise_predictor_diff_tuning, noise_schedule)
# x_out = dpm_solver_diff_tuning.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out, true_labels=target_labels_plot, diffused_labels=None,
#                        title_true_data="True Target Distribution", title_diffused_data="Diff-Tuning Sampled Distribution",
#                        save_name=f"results/enhance_net/diff_tuning_n{args.n}")
# print(f"Diff-Tuning plot saved to: results/enhance_net/diff_tuning_n{args.n}.png")

# # Calculate likelihood for Diff-Tuning (commented out for cosine modulation)
# # pdf_vals_diff_tuning = mixture_pdf_bimodal(x_out, mu_T_main, mu_T_side, sigma2_main, sigma2_side, w_main)
# # avg_likelihood_diff_tuning = pdf_vals_diff_tuning.mean().item()
# # print(f"Diff-Tuning average likelihood: {avg_likelihood_diff_tuning:.6f}")
avg_likelihood_diff_tuning = 0.0

# ====================================================================
# ====================================================================
# SCHEME 7: DIFF-TUNING WITH RATIO (KNOWLEDGE RETENTION + RECONSOLIDATION) - COMMENTED OUT
# ====================================================================
# print("\n" + "="*80)
# print("SCHEME 7: DIFF-TUNING WITH RATIO (KNOWLEDGE RETENTION + RECONSOLIDATION)")
# print("="*80)
# print("Description: Combined Knowledge Retention and Knowledge Reconsolidation with Ratio")
# print("Training: L_retention(θ) + L_adaptation(θ)")
# print("Retention: ratio^t * loss on source data (ratio from time-dependent classifier)")
# print("Adaptation: 1 * loss on target data (fixed coefficient)")
# print("="*80)

# # Train noise predictor with Diff-Tuning Ratio using both source and target data
# noise_predictor_diff_tuning_ratio = NoisePredictor()
# train_diffusion_noise_prediction_diff_tuning_ratio(
#     noise_predictor_diff_tuning_ratio, source_data, target_data_, classifier_balanced, 
#     n_epochs=source_epochs, batch_size=batch_size, lr=learning_rate, device=device
# )

# # Sample from the Diff-Tuning Ratio trained model (no guidance needed)
# dpm_solver_diff_tuning_ratio = DPM_Solver(noise_predictor_diff_tuning_ratio, noise_schedule)
# x_out = dpm_solver_diff_tuning_ratio.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out, true_labels=target_labels_plot, diffused_labels=None,
#                        title_true_data="True Target Distribution", title_diffused_data="Diff-Tuning Ratio Sampled Distribution",
#                        save_name=f"results/enhance_net/diff_tuning_ratio_n{args.n}")
# print(f"Diff-Tuning Ratio plot saved to: results/enhance_net/diff_tuning_ratio_n{args.n}.png")

# # Calculate likelihood for Diff-Tuning Ratio (commented out for cosine modulation)
# # pdf_vals_diff_tuning_ratio = mixture_pdf_bimodal(x_out, mu_T_main, mu_T_side, sigma2_main, sigma2_side, w_main)
# # avg_likelihood_diff_tuning_ratio = pdf_vals_diff_tuning_ratio.mean().item()
# # print(f"Diff-Tuning Ratio average likelihood: {avg_likelihood_diff_tuning_ratio:.6f}")
avg_likelihood_diff_tuning_ratio = 0.0

# ====================================================================
# SCHEME 8: DIFF-TUNING RATIO V3 (KNOWLEDGE RETENTION + RECONSOLIDATION) - COMMENTED OUT
# ====================================================================
# print("\n" + "="*80)
# print("SCHEME 8: DIFF-TUNING RATIO V3 (KNOWLEDGE RETENTION + RECONSOLIDATION)")
# print("="*80)
# print("Description: Combined Knowledge Retention and Knowledge Reconsolidation with Ratio V3")
# print("Training: L_retention(θ) + L_adaptation(θ)")
# print("Retention: ratio * loss on source data (ratio directly, not raised to power t)")
# print("Adaptation: 1 * loss on target data (fixed coefficient)")
# print("="*80)

# # Train noise predictor with Diff-Tuning Ratio V3 using both source and target data
# noise_predictor_diff_tuning_ratio_v3 = NoisePredictor()
# train_diffusion_noise_prediction_diff_tuning_ratio_v3(
#     noise_predictor_diff_tuning_ratio_v3, source_data, target_data_, classifier_balanced, 
#     n_epochs=source_epochs, batch_size=batch_size, lr=learning_rate, device=device
# )

# # Sample from the Diff-Tuning Ratio V3 trained model (no guidance needed)
# dpm_solver_diff_tuning_ratio_v3 = DPM_Solver(noise_predictor_diff_tuning_ratio_v3, noise_schedule)
# x_out = dpm_solver_diff_tuning_ratio_v3.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# plot_true_and_diffused(true_data=target_data_plot, diffused_data=x_out, true_labels=target_labels_plot, diffused_labels=None,
#                        title_true_data="True Target Distribution", title_diffused_data="Diff-Tuning Ratio V3 Sampled Distribution",
#                        save_name=f"results/enhance_net/diff_tuning_ratio_v3_n{args.n}")
# print(f"Diff-Tuning Ratio V3 plot saved to: results/enhance_net/diff_tuning_ratio_v3_n{args.n}.png")

# # Calculate likelihood for Diff-Tuning Ratio V3 (commented out for cosine modulation)
# # pdf_vals_diff_tuning_ratio_v3 = mixture_pdf_bimodal(x_out, mu_T_main, mu_T_side, sigma2_main, sigma2_side, w_main)
# # avg_likelihood_diff_tuning_ratio_v3 = pdf_vals_diff_tuning_ratio_v3.mean().item()
# # print(f"Diff-Tuning Ratio V3 average likelihood: {avg_likelihood_diff_tuning_ratio_v3:.6f}")
avg_likelihood_diff_tuning_ratio_v3 = 0.0

# # Scheme 8: Classifier-based guidance with regularization
# print("\n" + "="*60)
# print("SCHEME 3: CLASSIFIER-BASED GUIDANCE WITH REGULARIZATION")
# print("="*60)

# # Reuse the previously trained classifier (no need to retrain)
# classifier_reg = classifier
# print("Reusing previously trained classifier for regularization")

# # Train time-dependent classifier for regularization (using same data as classifier)
# classifier_time_dependent_reg = Classifier(input_dim=d+1)
# classifier_time_dependent_reg = train_time_dependent_classifier(classifier_time_dependent_reg, source_data, target_data_, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)
# print("Training time-dependent classifier with all source data (same as classifier)")

# # Train guidance network with regularization
# guidance_network_classifier_reg = GuidanceNetwork()
# guidance_network_classifier_reg = train_guidance_network_with_regularization(guidance_network_classifier_reg, classifier_reg, classifier_time_dependent_reg, noise_predictor_tgdp, noise_schedule, source_data, target_data_,
#                                                                   eta1and2=[1.0, 1.0], T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)

# # Test classifier-based guidance with regularization
# dpm_solver_classifier_reg = DPM_Solver(model_fn=noise_predictor_tgdp, guidance_scale=guidance_scale, noise_schedule=noise_schedule, guidance_network=guidance_network_classifier_reg)
# x_out_classifier_reg = dpm_solver_classifier_reg.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)

# # Determine the scheme name for classifier-based guidance with regularization
# scheme_name_classifier_reg = f"guidance_classifier_eta1.0_1.0_n{args.n}"

# plot_true_and_diffused(true_data=target_data_, diffused_data=x_out_classifier_reg, true_labels=target_labels_all_n[n_target_index], diffused_labels=None
#                        , title_true_data="Target Distribution", title_diffused_data=f"Classifier-based Guidance with Regularization",
#                        save_name=scheme_name_classifier_reg)

# pdf_vals_guidance_classifier_reg = mixture_pdf(x_out_classifier_reg, mu_T, sigma2)
# avg_likelihood_guidance_classifier_reg = pdf_vals_guidance_classifier_reg.mean().item()
# print(f"Classifier-based guidance with regularization average likelihood: {avg_likelihood_guidance_classifier_reg:.6f}")

# # Scheme 4: Balanced training guidance with regularization
# print("\n" + "="*60)
# print("SCHEME 4: BALANCED TRAINING GUIDANCE WITH REGULARIZATION")
# print("="*60)

# # Reuse the previously trained balanced classifier (no need to retrain)
# classifier_balanced_reg = classifier_balanced
# print("Reusing previously trained balanced classifier for regularization")

# # Reuse the previously trained balanced time-dependent classifier (no need to retrain)
# classifier_time_dependent_balanced_reg = classifier_time_dependent_balanced
# print("Reusing previously trained balanced time-dependent classifier for regularization (trained with balanced data)")

# # Train guidance network with balanced regularization
# guidance_network_balanced_reg = GuidanceNetwork()
# guidance_network_balanced_reg = train_guidance_network_with_regularization(guidance_network_balanced_reg, classifier_balanced_reg, classifier_time_dependent_balanced_reg, noise_predictor_tgdp, noise_schedule, source_data, target_data_,
#                                                                   eta1and2=[0.1, 0.1], T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)

# class DPM_SolverSafe:
#     """
#     1) 采样阶段默认 no_grad()，避免构建巨型计算图；
#     2) 只有在计算 ∇_x log h_psi 时短暂启用梯度；
#     3) 采样按小批次分块，避免 1 万大批次一次性爆内存。
#     """
#     def __init__(self, model_fn, noise_schedule, guidance_scale=[0.0, 0.0],
#                  guidance_network=None, sample_batch_size=1024):
#         # model_fn: (x, t_scalar_batch) -> predicted noise, 期望无梯度调用
#         self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
#         self.noise_schedule = noise_schedule
#         self.guidance_network = guidance_network  # 期望是 GuidanceNetworkPos
#         self.scale1 = guidance_scale[0]
#         self.scale2 = guidance_scale[1]
#         self.sample_batch_size = sample_batch_size

#     def _grad_log_h(self, x, t_scalar):
#         """
#         仅对 x 启用梯度，计算 ∇_x log h_psi(x,t)
#         返回 detached 的梯度张量。
#         """
#         with torch.enable_grad():
#             x = x.detach().requires_grad_(True)
#             t_full = t_scalar.expand(x.shape[0], 1)
#             h_pos = self.guidance_network(torch.cat([x, t_full], dim=1))  # > 0
#             log_sum = torch.log(h_pos).sum()
#             grad = torch.autograd.grad(log_sum, x, create_graph=False)[0]
#         return grad.detach()

#     def _first_order_update(self, x, s, t):
#         ns = self.noise_schedule
#         lam_s = ns.marginal_lambda(s)
#         lam_t = ns.marginal_lambda(t)
#         h = lam_t - lam_s
#         phi_1 = torch.expm1(h)

#         # denoiser 在 no_grad 环境被调用
#         model_s_source = self.model(x, s)

#         if self.guidance_network is not None:
#             guidance_s = -ns.marginal_std(s) * self._grad_log_h(x, s)
#             model_s_source = model_s_source + self.scale1 * guidance_s

#         x_t = (torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)) * x
#                - ns.marginal_std(t) * phi_1 * model_s_source)
#         return x_t

#     def _second_order_update(self, x, s, t):
#         ns = self.noise_schedule
#         lam_s = ns.marginal_lambda(s)
#         lam_t = ns.marginal_lambda(t)
#         h = lam_t - lam_s
#         r1 = 0.5
#         lam_s1 = lam_s + r1 * h
#         s1 = ns.inverse_lambda(lam_s1)

#         # step 1: at s
#         model_s = self.model(x, s)
#         if self.guidance_network is not None:
#             g_s = -ns.marginal_std(s) * self._grad_log_h(x, s)
#             model_s = model_s + self.scale1 * g_s

#         # go to s1
#         phi_11 = torch.expm1(r1 * h)
#         x_s1 = (torch.exp(ns.marginal_log_mean_coeff(s1) - ns.marginal_log_mean_coeff(s)) * x
#                 - ns.marginal_std(s1) * phi_11 * model_s)

#         # step 2: at s1
#         model_s1 = self.model(x_s1, s1)
#         if self.guidance_network is not None:
#             g_s1 = -ns.marginal_std(s1) * self._grad_log_h(x_s1, s1)
#             model_s1 = model_s1 + self.scale2 * g_s1

#         phi_1 = torch.expm1(h)
#         x_t = (torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)) * x
#                - ns.marginal_std(t) * phi_1 * model_s
#                - 0.5 * (ns.marginal_std(t) * phi_1) * (model_s1 - model_s))
#         return x_t

#     @torch.no_grad()
#     def second_order_sample_guidance(self, x, steps=25, t_start=1.0, t_end=1e-3):
#         """
#         采样全程 no_grad；仅内部 _grad_log_h 会短暂启用梯度。
#         x: [N, d] 初始 N(0,I) 噪声
#         """
#         x = x.to('cpu')
#         outs = []
#         for xb in x.split(self.sample_batch_size):
#             ts = torch.linspace(t_start, t_end, steps+1)  # cpu 上的标量时间
#             s = ts[0].unsqueeze(0)
#             x_cur = xb
#             for step in range(1, steps+1):
#                 t = ts[step].unsqueeze(0)
#                 if step == 1:
#                     x_cur = self._first_order_update(x_cur, s, t)
#                 else:
#                     x_cur = self._second_order_update(x_cur, s, t)
#                 s = t
#             outs.append(x_cur)
#         return torch.cat(outs, dim=0)

# # dpm_solver_balanced_reg = DPM_Solver(model_fn=noise_predictor_tgdp, guidance_scale=guidance_scale, noise_schedule=noise_schedule, guidance_network=guidance_network_balanced_reg)
# # x_out_balanced_reg = dpm_solver_balanced_reg.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
# dpm_solver_balanced_reg = DPM_SolverSafe(
#     model_fn=noise_predictor_tgdp,
#     noise_schedule=noise_schedule,
#     guidance_scale=guidance_scale,
#     guidance_network=guidance_network_balanced_reg,
#     sample_batch_size=1024   # 或 512
# )
# x_out_balanced_reg = dpm_solver_balanced_reg.second_order_sample_guidance(
#     x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3
# )


# Determine the scheme name for balanced guidance with regularization
scheme_name_balanced_reg = f"guidance_balanced_eta0.1_0.1_n{args.n}"

# plot_true_and_diffused(true_data=target_data_, diffused_data=x_out_balanced_reg, true_labels=target_labels_all_n[n_target_index], diffused_labels=None
#                        , title_true_data="Target Distribution", title_diffused_data=f"Balanced Training Guidance with Regularization",
#                        save_name=scheme_name_balanced_reg)

# pdf_vals_guidance_balanced_reg = mixture_pdf(x_out_balanced_reg, mu_T, sigma2)
# avg_likelihood_guidance_balanced_reg = pdf_vals_guidance_balanced_reg.mean().item()
# print(f"Balanced training guidance with regularization average likelihood: {avg_likelihood_guidance_balanced_reg:.6f}")

noise_predictor_source.to('cpu')

# -----------------------------------------------------------------------
# Summary: Compare all methods' average likelihood
# -----------------------------------------------------------------------
print("\n" + "="*90)
print("AVERAGE LIKELIHOOD COMPARISON - ALL SCHEMES")
print("="*90)
if show_baseline:
    print(f"1. Vanilla Diffusion:                           {avg_likelihood_vanilla:.6f}")
    print(f"2. Fine-tuned Target:                           {avg_likelihood_finetuned:.6f}")
    print(f"3. Classifier-based Guidance (no reg):          {avg_likelihood_guidance_classifier:.6f}")
    print(f"4. Balanced Training Guidance (no reg):         {avg_likelihood_guidance_balanced:.6f}")
    print(f"5. Multinomial Resampling:                      {avg_likelihood_multinomial:.6f}")
    print(f"6. OT Resampling:                               {avg_likelihood_ot:.6f}")
    # print(f"7. Importance Sampling (time-dependent):        {avg_likelihood_is_unbalanced:.6f}")  # COMMENTED OUT
    # print(f"8. Importance Sampling (balanced time-dep):     {avg_likelihood_is_balanced:.6f}")  # COMMENTED OUT
    # print(f"9. Importance Sampling (t-power):              {avg_likelihood_is_t_power:.6f}")  # COMMENTED OUT
    # print(f"7. Diff-Tuning (retention + reconsolidation):   {avg_likelihood_diff_tuning:.6f}")  # COMMENTED OUT
    # print(f"8. Diff-Tuning Ratio (ratio^t + fixed):         {avg_likelihood_diff_tuning_ratio:.6f}")  # COMMENTED OUT
    # print(f"9. Diff-Tuning Ratio V3 (ratio + fixed):         {avg_likelihood_diff_tuning_ratio_v3:.6f}")  # COMMENTED OUT
else:
    print("Baseline methods skipped (use --show_baseline to include them)")
    print(f"1. Classifier-based Guidance (no reg):          {avg_likelihood_guidance_classifier:.6f}")
    print(f"2. Balanced Training Guidance (no reg):         {avg_likelihood_guidance_balanced:.6f}")
    print(f"3. Multinomial Resampling:                      {avg_likelihood_multinomial:.6f}")
    print(f"4. OT Resampling:                               {avg_likelihood_ot:.6f}")
    # print(f"5. Importance Sampling (time-dependent):        {avg_likelihood_is_unbalanced:.6f}")  # COMMENTED OUT
    # print(f"6. Importance Sampling (balanced time-dep):     {avg_likelihood_is_balanced:.6f}")  # COMMENTED OUT
    # print(f"7. Importance Sampling (t-power):              {avg_likelihood_is_t_power:.6f}")  # COMMENTED OUT
    # print(f"5. Diff-Tuning (retention + reconsolidation):   {avg_likelihood_diff_tuning:.6f}")  # COMMENTED OUT
    # print(f"6. Diff-Tuning Ratio (ratio^t + fixed):         {avg_likelihood_diff_tuning_ratio:.6f}")  # COMMENTED OUT
    # print(f"7. Diff-Tuning Ratio V3 (ratio + fixed):         {avg_likelihood_diff_tuning_ratio_v3:.6f}")  # COMMENTED OUT
# print(f"8. Classifier-based Guidance (with reg):       {avg_likelihood_guidance_classifier_reg:.6f}")
# print(f"9. Balanced Training Guidance (with reg):     {avg_likelihood_guidance_balanced_reg:.6f}") 
print("="*90)
avg_likelihood_pretrained = 0.0  # Set default value
print(f"Pre-trained Source (finetune step):             {avg_likelihood_pretrained:.6f}")
print("="*90)

# Find the best performing method
likelihoods = {
    "Classifier-based Guidance (no reg)": avg_likelihood_guidance_classifier,
    "Balanced Training Guidance (no reg)": avg_likelihood_guidance_balanced,
    "Multinomial Resampling": avg_likelihood_multinomial,
    "OT Resampling": avg_likelihood_ot,
    # "Importance Sampling (time-dependent)": avg_likelihood_is_unbalanced,  # COMMENTED OUT
    # "Importance Sampling (balanced time-dep)": avg_likelihood_is_balanced,  # COMMENTED OUT
    # "Importance Sampling (t-power)": avg_likelihood_is_t_power,  # COMMENTED OUT
    # "Diff-Tuning (retention + reconsolidation)": avg_likelihood_diff_tuning,  # COMMENTED OUT
    # "Diff-Tuning Ratio (ratio^t + fixed)": avg_likelihood_diff_tuning_ratio,  # COMMENTED OUT
    # "Diff-Tuning Ratio V3 (ratio + fixed)": avg_likelihood_diff_tuning_ratio_v3,  # COMMENTED OUT
    # "Classifier-based Guidance (with reg)": avg_likelihood_guidance_classifier_reg,
    # "Balanced Training Guidance (with reg)": avg_likelihood_guidance_balanced_reg
}

# Add baseline methods only if they were run
if show_baseline:
    likelihoods["Vanilla Diffusion"] = avg_likelihood_vanilla
    likelihoods["Fine-tuned Target"] = avg_likelihood_finetuned

best_method = max(likelihoods, key=likelihoods.get)
best_likelihood = likelihoods[best_method]

print(f"\nBEST PERFORMING METHOD: {best_method}")
print(f"BEST LIKELIHOOD: {best_likelihood:.6f}")
print("="*90)

# Additional analysis: Compare regularization effects
print("\n" + "="*60)
print("REGULARIZATION EFFECT ANALYSIS")
print("="*60)
# classifier_improvement = avg_likelihood_guidance_classifier_reg - avg_likelihood_guidance_classifier
# balanced_improvement = avg_likelihood_guidance_balanced_reg - avg_likelihood_guidance_balanced


# -----------------------------------------------------------------------
# 9. Evaluate the Learned Density Ratio (Figure 2)
# -----------------------------------------------------------------------
if show_density_ratio:
    classifier.to('cpu')
    classifier_balanced.to('cpu')
    # Commented out density ratio comparison for cosine modulation example
    # This section was specific to Gaussian distributions
    # N = 1000
    # # Generate samples from target distribution (bimodal)
    # t_rand_main = generate_samples(mu_T_main, sigma2_main, int(10000 * w_main))
    # t_rand_side = generate_samples(mu_T_side, sigma2_side, int(10000 * (1 - w_main)))
    # t_rand = torch.cat([t_rand_main[0], t_rand_side[0]], dim=0)
    # 
    # s_rand = generate_samples(mu_S, sigma2_source, 10000)
    # x_rand = torch.cat([t_rand, s_rand[0]], dim=0)
    # 
    # 
    # with torch.no_grad():
    #     # Oracle log-ratio for bimodal target
    #     ratio_oracle = true_ratio_oracle_bimodal(x_rand, mu_S, mu_T_main, mu_T_side, sigma2_source, sigma2_main, sigma2_side, w_main)
    #     log_ratio_oracle = torch.log(ratio_oracle + 1e-20)
    print("Density ratio comparison commented out for cosine modulation example")
    
    # # Learned log-ratio from unbalanced classifier
    # cvals_unbalanced = classifier(x_rand)  # shape [N,1]
    # cvals_unbalanced_clamped = torch.clamp(cvals_unbalanced, min=1e-8, max=1-1e-8)
    # ratio_learned_unbalanced = (1 - cvals_unbalanced_clamped) / cvals_unbalanced_clamped
    # ratio_learned_unbalanced = torch.clamp(ratio_learned_unbalanced, min=1e-8, max=1e8)
    # log_ratio_learned_unbalanced = torch.log(ratio_learned_unbalanced + 1e-20)
    # 
    # # Learned log-ratio from balanced classifier
    # cvals_balanced = classifier_balanced(x_rand)  # shape [N,1]
    # cvals_balanced_clamped = torch.clamp(cvals_balanced, min=1e-8, max=1-1e-8)
    # ratio_learned_balanced = (1 - cvals_balanced_clamped) / cvals_balanced_clamped
    # ratio_learned_balanced = torch.clamp(ratio_learned_balanced, min=1e-8, max=1e8)
    # log_ratio_learned_balanced = torch.log(ratio_learned_balanced + 1e-20)
    # 
    # # Print ratio statistics for comparison
    # print(f"\n=== DENSITY RATIO STATISTICS ===")
    # print(f"Ratio stats - Oracle: {ratio_oracle.mean().item():.4f}, Unbalanced: {ratio_learned_unbalanced.mean().item():.4f}, Balanced: {ratio_learned_balanced.mean().item():.4f}")
    # print(f"================================\n")
    # 
    # # Plot with three subplots: Oracle, Unbalanced, Balanced
    # plot_ratio_comparison(x_rand, log_ratio_oracle, log_ratio_learned_unbalanced, log_ratio_learned_balanced, 
    #                      save_name=f"results/IS/density_ratio_comparison_n{args.n}")
    # print(f"Density ratio comparison plot saved to: results/IS/density_ratio_comparison_n{args.n}.png")




