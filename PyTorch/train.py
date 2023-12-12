# シードの設定を行う関数
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# モデルサイズを計算する関数
def calculate_model_size(model):
    total_size = 0
    for param in model.parameters():
        param_size = param.numel()
        total_size += param_size * param.element_size()

    total_size_bytes = total_size
    total_size_kb = total_size / 1024
    total_size_mb = total_size_kb / 1024
    total_size_gb = total_size_mb / 1024

    print(f"Model size: {total_size_bytes} bytes / {total_size_kb:.2f} KB / {total_size_mb:.2f} MB / {total_size_gb:.4f} GB")

# ダウンサンプリングを行うための関数
def create_balanced_sampler(dataset):
    label_counts = Counter()
    for _, (current_label, next_label) in dataset:
        label_counts[current_label.item()] += 1
        label_counts[next_label.item()] += 1

    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    weights = {label: 1.0 / count for label, count in label_counts.items()}
    
    sample_weights = [weights[current_label.item()] + weights[next_label.item()] for _, (current_label, next_label) in dataset]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler