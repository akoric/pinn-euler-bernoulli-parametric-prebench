import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import ParametricClampedBeam, analytical_solution

# generate training data with importance sampling
def gen_data(n=15000, E_range=(100e9,300e9), I_range=(0.5e-6,2e-6), q_range=(500,1500), L=1.0,
             importance=True, hi_I=(1.3e-6,2e-6), hi_q=(1200.0,1500.0)):
    if not importance:
        E=np.random.uniform(*E_range, n)
        I=np.random.uniform(*I_range, n)
        q=np.random.uniform(*q_range, n)
    else:
        # split data into different regions for better coverage
        n_uni=n//2
        n_hiI=n//4
        n_hiq=n - n_uni - n_hiI
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
    x=np.random.beta(0.5,0.5,n)  # beta dist samples more at boundaries
    w=np.array([analytical_solution(xi,L,Ei,Ii,qi) for xi,Ei,Ii,qi in zip(x,E,I,q)])
    return {
        'x':torch.tensor(x.reshape(-1,1),dtype=torch.float32),
        'E':torch.tensor(E.reshape(-1,1),dtype=torch.float32),
        'I':torch.tensor(I.reshape(-1,1),dtype=torch.float32),
        'q':torch.tensor(q.reshape(-1,1),dtype=torch.float32),
        'w':torch.tensor(w.reshape(-1,1),dtype=torch.float32),
    }

def pde_residual(E,I,q,out):
    eps=torch.finfo(out['w_xxxx'].dtype).eps
    return (E*I*out['w_xxxx'])/(q+eps) - 1.0

# main training loop
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L=1.0
model=ParametricClampedBeam([64,64,64],L=L).to(device)
model.set_normalization_params(200e9,1e-6,1000.0)

# set seed for reproducible data generation
torch.manual_seed(42)
np.random.seed(42)
data=gen_data(15000)
for k in data:
    data[k]=data[k].to(device)
    
# 80/20 train/validation split
n_total = data['x'].shape[0]
perm = torch.randperm(n_total, device=device)
n_train = int(0.8 * n_total)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

def compute_val_stats():
    """Return (rmse, rms_true, rel_percent) on the 20% validation split."""
    model.eval()
    x = data['x'][val_idx].clone().detach().requires_grad_(True)
    E = data['E'][val_idx]
    I = data['I'][val_idx]
    q = data['q'][val_idx]
    w_true = data['w'][val_idx]
    out = model(x, E, I, q)
    err = out['w'] - w_true
    rmse = torch.sqrt((err**2).mean())
    rms_true = torch.sqrt((w_true**2).mean())
    rel_percent = 100.0 * (rmse / (rms_true + 1e-15))
    return rmse.item(), rms_true.item(), rel_percent.item()

opt=optim.Adam(model.parameters(), lr=5e-4)
sched=optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=400)

# loss weights
pde_w, bc_w, data_w = 30.0, 100.0, 1.0
start=time.time()

# track loss history
loss_history = {
    'total': [],
    'pde': [],
    'bc': [],
    'data': [],
    'val_rmse': [],
    'val_rel': []
}

pbar=tqdm(range(6000), desc='Training')
for epoch in pbar:
    model.train()
    opt.zero_grad()
    # Sample only from training subset
    idx_in_tr = torch.randint(0, train_idx.shape[0], (512,), device=device)
    idx = train_idx[idx_in_tr]
    x=data['x'][idx].requires_grad_(True)
    E=data['E'][idx]
    I=data['I'][idx]
    q=data['q'][idx]
    w_true=data['w'][idx]
    out=model(x,E,I,q)
    
    res=pde_residual(E,I,q,out)
    loss_pde=(res**2).mean()
    
    # boundary conditions at x=1
    n_bc=64
    x_bc=torch.ones(n_bc,1,device=device,requires_grad=True)
    E_bc=torch.rand(n_bc,1,device=device)*200e9 + 100e9
    I_bc=torch.rand(n_bc,1,device=device)*1.5e-6 + 0.5e-6
    q_bc=torch.rand(n_bc,1,device=device)*1000 + 500
    out_bc=model(x_bc,E_bc,I_bc,q_bc)
    loss_bc=(out_bc['w_xx']**2).mean() + (out_bc['w_xxx']**2).mean()
    
    # Normalized shape data loss: scale by s = q*L^4/(E*I)
    s = (q * (L**4)) / (E * I + 1e-15)
    loss_data=(((out['w']/s) - (w_true/s))**2).mean()
    
    loss=pde_w*loss_pde + bc_w*loss_bc + data_w*loss_data
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step()
    sched.step(loss)
    
    # record losses every epoch
    loss_history['total'].append(loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['bc'].append(loss_bc.item())
    loss_history['data'].append(loss_data.item())
    
    if epoch%200==0:
        val_rmse, val_rms_true, val_rel = compute_val_stats()
        loss_history['val_rmse'].append(val_rmse)
        loss_history['val_rel'].append(val_rel)
        pbar.set_postfix({
            'Loss': f"{loss.item():.2e}",
            'PDE': f"{loss_pde.item():.2e}",
            'BC': f"{loss_bc.item():.2e}",
            'Data': f"{loss_data.item():.2e}",
            'ValRMSE': f"{val_rmse:.3e}",
            'Val%': f"{val_rel:.2f}%",
        })
        
print(f"Training took {(time.time()-start):.1f}s")
final_rmse, final_rms_true, final_rel = compute_val_stats()
print(f"Final validation RMSE: {final_rmse:.6e} m")
print(f"Validation RMS(w_true): {final_rms_true:.6e} m")
print(f"Validation Relative RMSE: {final_rel:.3f}%")

# save model with validation indices for reproducibility
torch.save({
    'model_state_dict':model.state_dict(),
    'E_ref':200e9,
    'I_ref':1e-6,
    'q_ref':1000.0,
    'L':L,
    'hidden_layers':[64,64,64],
    'val_idx':val_idx.cpu(),
    'data_seed':42,
    'n_data':15000,
    'loss_history':loss_history
}, 'parametric_pinn_model.pt')
print('Saved model to parametric_pinn_model.pt')
