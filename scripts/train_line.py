import torch
import torch.optim as optim
from dendritic_modeling.models import EINetClassifierNLL   
from synthetic_line_dataset import get_line_dataloaders
from torch.utils.data import DataLoader

def accuracy_score(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            logits_or_dist = model.forward(x)  # for MLP: raw logits; for EINet: Categorical dist or logits
            # If using EINetClassifierNLL, the forward(x) returns a Categorical distribution if y=None, sampler=True
            # or returns logits if sampler=False. Let's unify how we pick predicted labels:
            if hasattr(logits_or_dist, 'logits'):
                # This is a distribution
                preds = logits_or_dist.logits.argmax(dim=-1)
            else:
                # This is a raw tensor of shape (batch_size, num_classes)
                preds = logits_or_dist.argmax(dim=-1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def train_model(model, train_loader, valid_loader, epochs=10, lr=1e-3):
    # We unify MLP and EINet by using cross-entropy if MLP, or NLL if EINet
    # For EINetClassifierNLL, we might do a direct manual NLL. 
    # But let's just do a standard approach with cross entropy for MLP.
    
    is_dendritic = isinstance(model, EINetClassifierNLL)

    if is_dendritic:
        # The model output is already NLL if we do forward(x, y)
        # => We'll do manual update
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                # forward: returns nll for each sample
                nll = model.forward(x, y)  # shape [batch_size]
                loss = nll.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(train_loader.dataset)

            # Evaluate on validation
            val_acc = accuracy_score(model, valid_loader)
            print(f"[EINet] Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

    else:
        # For MLP, we'll do CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(train_loader.dataset)

            # Evaluate on validation
            val_acc = accuracy_score(model, valid_loader)
            print(f"[MLP] Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

def main():
    # 1. Data
    train_loader, valid_loader, test_loader = get_line_dataloaders(
        train_size=8000, valid_size=1000, test_size=1000,
        image_size=10, batch_size=64
    )
    input_dim = 10 * 10
    num_classes = 3

    # 2. Baseline MLP
    mlp_model = MLPBaseline(input_dim=input_dim, hidden_dim=50, output_dim=num_classes)
    print("\n=== Training Baseline MLP ===")
    train_model(mlp_model, train_loader, valid_loader, epochs=10, lr=1e-3)
    mlp_test_acc = accuracy_score(mlp_model, test_loader)
    print(f"Baseline MLP Test Accuracy: {mlp_test_acc:.4f}\n")

    # 3. Dendritic Model (EINetClassifierNLL)

    dendritic_model = EINetClassifierNLL(
        input_dim=input_dim,
        excitatory_layer_sizes=[10],     
        inhibitory_layer_sizes=[10],     
        excitatory_branch_factors=[2],    
        inhibitory_branch_factors=[2],
        ee_synapses_per_branch_per_layer=[10],
        ei_synapses_per_branch_per_layer=[10],
        ie_synapses_per_branch_per_layer=[1],
        ii_synapses_per_branch_per_layer=[1],
        reactivate=True,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
    )

    print("=== Training DendriNet (EINet) ===")
    train_model(dendritic_model, train_loader, valid_loader, epochs=10, lr=1e-3)
    einet_test_acc = accuracy_score(dendritic_model, test_loader)
    print(f"DendriNet (EINet) Test Accuracy: {einet_test_acc:.4f}\n")

if __name__ == "__main__":
    main()