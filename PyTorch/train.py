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

# 訓練のための関数
def train(opt):

    # シードの設定
    seed_everything(opt.seed)

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # オプションからパラメータの設定
    option = opt.option

    # Data preprocessing steps
    transform = Compose([
        # ここに前処理を追加
    ])

    # データセットの作成
    video_dataset = VideoDataset(csv_file=csv_file, transform=transform, addpath=addpath, usecache=cache)

    # データセットの分割
    train_size = int(0.7 * len(video_dataset))
    val_size = len(video_dataset) - train_size
    train_dataset, val_dataset = random_split(video_dataset, [train_size, val_size])

    # データのダウンサンプリングの適用
    train_sampler = create_balanced_sampler(train_dataset)
    val_sampler = create_balanced_sampler(val_dataset)

    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, sampler=train_sampler, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, sampler=val_sampler, num_workers=8)

    # Initializing the model
    model = CustomSlowFast(device=device, num_classes=2)

    # Utilizing multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Calculating model size
    calculate_model_size(model)

    # Setting up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    criterion = nn.CrossEntropyLoss()

    # Initialize minimum validation loss for early stopping
    val_loss_min = None
    val_loss_min_epoch = 0

    # Arrays to record the training and validation results
    train_losses = []
    val_losses = []
    val_accuracy = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        # Set model to training mode
        model.train()
        # Reset epoch's training and validation loss
        train_loss = 0.0
        val_loss = 0.0

        # Initialize confusion matrices for each task
        confusion_matrix_current = np.zeros((2, 2))
        confusion_matrix_next = np.zeros((2, 2)) 

        # Training phase
        for i, (inputs, (current_labels, next_labels)) in enumerate(train_loader):
            inputs = inputs.to(device)
            current_labels = current_labels.to(device)
            next_labels = next_labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and backward pass
            current_outputs, next_outputs = model(inputs)
            loss_current = criterion(current_outputs, current_labels)
            loss_next = criterion(next_outputs, next_labels)
            # Combine losses for multitasking
            loss = (loss_current + loss_next) / 2
            loss.backward()
            optimizer.step()
            # Accumulate the loss
            train_loss += loss.item()

        # Calculate loss per epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Set model to evaluation mode
        model.eval()

        # Initialize variables for accuracy calculation
        correct = 0
        total = 0

        # Validation phase
        with torch.no_grad():

            # Initialize lists to store true labels and predictions for each task
            all_current_labels = []
            all_next_labels = []
            all_predicted_current = []
            all_predicted_next = []

            for i, (inputs, (current_labels, next_labels)) in enumerate(val_loader):
                inputs = inputs.to(device)
                current_labels = current_labels.to(device)
                next_labels = next_labels.to(device)

                # Forward pass
                current_outputs, next_outputs = model(inputs)

                # Predicted class with highest probability
                _, predicted_current = torch.max(current_outputs, 1)
                _, predicted_next = torch.max(next_outputs, 1)

                # Accumulate validation losses for both tasks
                val_loss += criterion(current_outputs, current_labels).item()
                val_loss += criterion(next_outputs, next_labels).item()

                # Accumulate accuracy
                total += current_labels.size(0)
                correct += (predicted_current == current_labels).sum().item()
                total += next_labels.size(0)
                correct += (predicted_next == next_labels).sum().item()

                # Store predictions and labels
                all_current_labels.extend(current_labels.cpu().numpy())
                all_next_labels.extend(next_labels.cpu().numpy())
                all_predicted_current.extend(predicted_current.cpu().numpy())
                all_predicted_next.extend(predicted_next.cpu().numpy())
            
            accuracy = 100 * correct / total
                
        # Calculate validation accuracy and loss
        accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

        # Calculate and display confusion matrix for each task
        confusion_matrix_current = confusion_matrix(all_current_labels, all_predicted_current)
        confusion_matrix_next = confusion_matrix(all_next_labels, all_predicted_next)

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        print(f'Confusion Matrix for Current Task:\n{confusion_matrix_current}')
        print(f'Confusion Matrix for Next Task:\n{confusion_matrix_next}')
        sys.stdout.flush()

        # Check for early stopping
        if val_loss_min is None or val_loss < val_loss_min:
            # Save model if validation loss decreased
            model_save_name = f'./noncacheresult/lr{learning_rate}_ep{epochs}_pa{patience}.pt'
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
        elif (epoch - val_loss_min_epoch) >= patience:
            # Early stopping if no improvement in validation loss
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            sys.stdout.flush()
            break

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.title("Training and Validation CrossEntropy Loss")
    plt.title("Accuracy")
    plt.savefig(f'./noncachegraph/lr{learning_rate}_ep{epochs}_pa{patience}.png')

    # Return final training and validation loss
    return train_loss, val_loss_min

# Main block to execute the script
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print('start time:',start_time)
    sys.stdout.flush()

    # Argument parser for command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/labels.csv', help='path to data file')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr_search', type=str, default='false', help='whether to perform learning rate search')
    parser.add_argument('--usecache', type=str, default='false', help='device to use for training')
    opt = parser.parse_args()

    # Learning rate search if specified
    if opt.lr_search == 'true':
        # Perform learning rate search
        learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        best_loss = float('inf')
        best_lr = 0
        train_losses = []
        val_losses = []
        for lr in learning_rates:
            opt.lr = lr
            print(f"\nTraining with learning rate: {lr}")
            sys.stdout.flush()
            print('-----beginning training-----')
            sys.stdout.flush()

            train_loss, val_loss = train(opt)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_lr = lr
        print('best validation loss: ', best_loss)
        sys.stdout.flush()
        print(f"Best learning rate: {best_lr}")
        sys.stdout.flush()
        print('Learning rate search results:')
        for lr, train_loss, val_loss in zip(learning_rates, train_losses, val_losses):
            print(f'Learning rate: {lr}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
    else:
        # Train the model with the given hyperparameters.
        print('-----biginning training-----')
        sys.stdout.flush()
        train_loss, val_loss = train(opt)
        print('final train loss: ',train_loss)
        print('final validation loss: ', val_loss)
        sys.stdout.flush()

    # Print script execution time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print('-----completing training-----')
    sys.stdout.flush()
    print('end time:',end_time)
    sys.stdout.flush()
    print('Execution time: ', execution_time)
    sys.stdout.flush()