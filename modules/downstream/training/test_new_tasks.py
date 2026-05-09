import os, sys
import torch

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from EHR_model import EHRModel, EHRLoss, EHRTransformer

def test_models():
    print("Testing EHRModel...")
    model = EHRModel(n_diagnoses=200, n_drugs=50)
    batch = {
        'emb': torch.randn(4, 10, 128),
        'dt': torch.randn(4, 10),
        'lengths': torch.tensor([10, 8, 5, 2]),
        'patient_vec': torch.randn(4, 64),
        'admission_vec': torch.randn(4, 64)
    }
    out = model(batch)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
        assert not torch.isnan(v).any()

    print("\nTesting EHRTransformer...")
    model_t = EHRTransformer(n_diagnoses=200, n_drugs=50)
    out_t = model_t(batch)
    for k, v in out_t.items():
        print(f"  {k}: {v.shape}")
        assert not torch.isnan(v).any()

    print("\nTesting EHRLoss...")
    batch_labels = {
        'mortality': torch.tensor([0., 1., 0., 1.]),
        'los_7d': torch.tensor([1., 0., 1., 0.]),
        'readmission': torch.tensor([1., -1., 0., 1.]),
        'progression': torch.zeros(4, 200),
        'drug_rec': torch.zeros(4, 50)
    }
    batch_labels['progression'][0, 10] = 1.0
    batch_labels['drug_rec'][0, 5] = 1.0
    
    # Add labels to batch
    batch.update(batch_labels)
    
    criterion = EHRLoss(
        pos_weight_mortality=torch.tensor([10.0]),
        pos_weight_los=torch.tensor([2.0]),
        pos_weight_readmission=torch.tensor([3.0]),
        pos_weight_progression=torch.ones(200),
        pos_weight_drug_rec=torch.ones(50)
    )
    
    loss, loss_dict = criterion(out, batch)
    print(f"Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    try:
        test_models()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
