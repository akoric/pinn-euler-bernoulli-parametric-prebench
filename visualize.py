import numpy as np
import torch
import matplotlib.pyplot as plt
from models import ParametricClampedBeam, analytical_solution

# copying from train.py (avoid importing it and dealing with dependencies)
def gen_data(n=15000, E_range=(100e9,300e9), I_range=(0.5e-6,2e-6), q_range=(500,1500), L=1.0,
             importance=True, hi_I=(1.3e-6,2e-6), hi_q=(1200.0,1500.0)):
    if not importance:
        E=np.random.uniform(*E_range, n)
        I=np.random.uniform(*I_range, n)
        q=np.random.uniform(*q_range, n)
    else:
        # split into different regions
        n_uni=n//2
        n_hiI=n//4
        n_hiq = n - n_uni - n_hiI
        E=np.concatenate([
            np.random.uniform(*E_range,n_uni),
            np.random.uniform(*E_range,n_hiI),
            np.random.uniform(*E_range,n_hiq),
        ])
        I=np.concatenate([
            np.random.uniform(*I_range,n_uni),
            np.random.uniform(*hi_I,n_hiI),
            np.random.uniform(*I_range,n_hiq),
        ])
        q=np.concatenate([
            np.random.uniform(*q_range,n_uni),
            np.random.uniform(*q_range,n_hiI),
            np.random.uniform(*hi_q,n_hiq),
        ])
    x=np.random.beta(0.5,0.5,n)
    w=np.array([analytical_solution(xi,L,Ei,Ii,qi) for xi,Ei,Ii,qi in zip(x,E,I,q)])
    
    return {
        'x':torch.tensor(x.reshape(-1,1),dtype=torch.float32),
        'E':torch.tensor(E.reshape(-1,1),dtype=torch.float32),
        'I':torch.tensor(I.reshape(-1,1),dtype=torch.float32),
        'q':torch.tensor(q.reshape(-1,1),dtype=torch.float32),
        'w':torch.tensor(w.reshape(-1,1),dtype=torch.float32),
    }

def load_model(path='parametric_pinn_model.pt'):
    ckpt=torch.load(path,map_location='cpu')
    model=ParametricClampedBeam(ckpt.get('hidden_layers',[64,64,64]), L=ckpt.get('L',1.0))
    model.load_state_dict(ckpt['model_state_dict'])
    model.set_normalization_params(ckpt['E_ref'], ckpt['I_ref'], ckpt['q_ref'])
    model.eval()
    return model, ckpt.get('L',1.0)

def plot_case(ax, model, L, E, I, q, label):
    x=torch.linspace(0,1,200).reshape(-1,1).requires_grad_(True)
    E_t=torch.full_like(x,float(E))
    I_t=torch.full_like(x,float(I))
    q_t=torch.full_like(x,float(q))
    out=model(x,E_t,I_t,q_t)
    
    w_p=out['w'].detach().numpy().flatten()*1000  # convert to mm
    x_np=x.detach().numpy().flatten()
    w_t=analytical_solution(x_np,L,float(E),float(I),float(q))*1000
    
    ax.plot(x_np,w_t,'k-',lw=2,label='Analytical')
    ax.plot(x_np,w_p,'r--',lw=2,label='PINN')
    ax.set_title(label)
    ax.set_xlabel('x/L')
    ax.set_ylabel('Deflection (mm)')
    ax.grid(True,alpha=0.3)

# Load the model
model,L=load_model()

# load checkpoint again to get more info
ckpt = torch.load('parametric_pinn_model.pt', map_location='cpu')
val_idx = ckpt.get('val_idx', None)
data_seed = ckpt.get('data_seed', 42)
n_data = ckpt.get('n_data', 15000)
loss_history = ckpt.get('loss_history', None)

# Regenerate validation data
if val_idx is None:
    print("Model checkpoint doesn't contain validation indices. Using default split.")
    torch.manual_seed(42)
    np.random.seed(42)
    data=gen_data(n_data)
    n_total = data['x'].shape[0]
    perm = torch.randperm(n_total)
    n_train = int(0.8 * n_total)
    val_idx = perm[n_train:]
else:
    # use same seed as training to reproduce exact validation set
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    data=gen_data(n_data)

# Extract validation data
x_val = data['x'][val_idx].requires_grad_(True)
E_val = data['E'][val_idx]
I_val = data['I'][val_idx]
q_val = data['q'][val_idx]
w_true_val = data['w'][val_idx]

# Run PINN on validation set
out = model(x_val, E_val, I_val, q_val)
w_pred_val = out['w'].detach()

w_true_mm = w_true_val.numpy().flatten() * 1000
w_pred_mm = w_pred_val.numpy().flatten() * 1000


# Plot loss curves if we have them
if loss_history is not None:
    print("Plotting training loss history...")
    fig_loss = plt.figure(figsize=(14, 10))
    
    # plot 1: total loss
    ax1 = plt.subplot(3, 2, 1)
    ax1.semilogy(loss_history['total'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Training Loss', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # plot 2: pde loss
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogy(loss_history['pde'], 'r-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PDE Loss')
    ax2.set_title('PDE Residual Loss', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # plot 3: bc loss
    ax3 = plt.subplot(3, 2, 3)
    ax3.semilogy(loss_history['bc'], 'g-', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('BC Loss')
    ax3.set_title('Boundary Condition Loss', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # plot 4: data fitting loss
    ax4 = plt.subplot(3, 2, 4)
    ax4.semilogy(loss_history['data'], 'm-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Data Loss')
    ax4.set_title('Data Fitting Loss', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # plot 5: validation RMSE over time
    ax5 = plt.subplot(3, 2, 5)
    # validation metrics were computed every 200 epochs
    val_epochs = list(range(0, len(loss_history['total']), 200))[:len(loss_history['val_rmse'])]
    ax5.semilogy(val_epochs, loss_history['val_rmse'], 'o-', linewidth=2, markersize=4, color='purple')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Validation RMSE (m)')
    ax5.set_title('Validation RMSE', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    # plot 6: relative error
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(val_epochs, loss_history['val_rel'], 'o-', linewidth=2, markersize=4, color='orange')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Relative Error (%)')
    ax6.set_title('Validation Relative Error', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=200, bbox_inches='tight')
    print('Saved training_loss.png')
    plt.close()


# main visualization plot
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

#scatter plot - analytical vs PINN predictions
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_main.scatter(w_true_mm, w_pred_mm, alpha=0.3, s=10, c='blue', edgecolors='none')

min_val = min(w_true_mm.min(), w_pred_mm.min())
max_val = max(w_true_mm.max(), w_pred_mm.max())
ax_main.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

ax_main.set_xlabel('Analytical Deflection (mm)', fontsize=11)
ax_main.set_ylabel('PINN Predicted Deflection (mm)', fontsize=11)
ax_main.set_title('Validation Set: Analytical vs PINN (3000 points)', fontsize=12, fontweight='bold')
ax_main.grid(True, alpha=0.3)
ax_main.legend()
ax_main.set_aspect('equal', adjustable='box')

# display error metrics
rmse = np.sqrt(((w_pred_mm - w_true_mm)**2).mean())
rel_error = 100 * rmse / np.sqrt((w_true_mm**2).mean())
ax_main.text(0.05, 0.95, f'RMSE: {rmse:.3f} mm\nRel Error: {rel_error:.2f}%', 
             transform=ax_main.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

# few example cases with different parameters
cases=[
    ('E=150 GPa', 150e9, 1e-6, 1000),
    ('I=1.5e-6', 200e9, 1.5e-6, 1000),
    ('q=1250', 200e9, 1e-6, 1250),
    ('E=300 (edge)', 300e9, 1e-6, 1000),
]

positions = [gs[0, 2], gs[1, 2], gs[2, 0], gs[2, 1]]
for pos, (name, E, I, q) in zip(positions, cases):
    ax = fig.add_subplot(pos)
    x=torch.linspace(0,1,200).reshape(-1,1).requires_grad_(True)
    E_t=torch.full_like(x,float(E))
    I_t=torch.full_like(x,float(I))
    q_t=torch.full_like(x,float(q))
    out=model(x,E_t,I_t,q_t)
    
    w_p=out['w'].detach().numpy().flatten()*1000
    x_np=x.detach().numpy().flatten()
    w_t=analytical_solution(x_np,L,float(E),float(I),float(q))*1000
    
    ax.plot(x_np,w_t,'k-',lw=1.5,label='Analytical')
    ax.plot(x_np,w_p,'r--',lw=1.5,label='PINN')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('x/L', fontsize=9)
    ax.set_ylabel('w (mm)', fontsize=9)
    ax.grid(True,alpha=0.3)
    ax.tick_params(labelsize=8)

# legend
ax_legend = fig.add_subplot(gs[2, 2])
ax_legend.axis('off')
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc='center', frameon=True, fontsize=11)
ax_legend.text(0.5, 0.2, 'Example Cases', ha='center', fontsize=10, fontweight='bold')

plt.savefig('visualizations.png', dpi=200, bbox_inches='tight')
print('Saved visualizations.png')
print("Done!")
