"""Review check 4: the early-stopping 'restore best weights' bug (state_dict() returns live references) reproduced on toy data."""
import copy, torch
torch.manual_seed(0)
X=torch.randn(60,5); y=X[:,0]+0.5*torch.randn(60); Xv=torch.randn(40,5); yv=Xv[:,0]+0.5*torch.randn(40)
def run(copy_state):
    torch.manual_seed(1); net=torch.nn.Sequential(torch.nn.Linear(5,64),torch.nn.ReLU(),torch.nn.Linear(64,1)); opt=torch.optim.Adam(net.parameters(),1e-2)
    best=float("inf"); best_state=None; since=0; hist=[]
    for ep in range(400):
        opt.zero_grad(); torch.nn.functional.mse_loss(net(X).squeeze(1),y).backward(); opt.step()
        with torch.no_grad(): v=torch.nn.functional.mse_loss(net(Xv).squeeze(1),yv).item()
        if v<best: best,since,best_ep=v,0,ep; best_state=copy.deepcopy(net.state_dict()) if copy_state else net.state_dict()
        else:
            since+=1
            if since>=30: break
    net.load_state_dict(best_state)
    with torch.no_grad(): after=torch.nn.functional.mse_loss(net(Xv).squeeze(1),yv).item()
    return best,best_ep,ep,after
for flag in (False,True):
    b,be,e,a=run(flag); print(f"deepcopy={flag!s:5}: best val MSE {b:.4f} at epoch {be}, stopped at {e}; val MSE after 'load best' = {a:.4f}  -> restore {'WORKED' if abs(a-b)<1e-6 else 'was a NO-OP (weights stayed at the last epoch)'}")
