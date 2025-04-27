import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1*28*28) / 255.

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:1000]
valid_labs = train_labs[:1000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

learning_rates = [0.002, 0.005, 0.01, 0.015, 0.02, 0.05]

results = {}

for lr in learning_rates:
    model = nn.models.Model_MLP([train_imgs.shape[-1], 1024, 512, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
    optimizer = nn.optimizer.Adam(init_lr=lr, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=None, batch_size=128)
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./saved_models', model_name=f'model_{lr}.pickle')

    results[lr] = runner

    _, axes = plt.subplots(1, 2)
    axes.reshape(-1)
    _.set_tight_layout(1)
    plot(runner, axes)
    plt.savefig(f'train_lr_{lr}.png')
    plt.close()

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

accuracies = []

for lr, runner in results.items():
    model.load_model(r'.\best_models\best_model.pickle')
    logits = model(test_imgs)
    accuracy = nn.metric.accuracy(logits, test_labs)
    accuracies.append(accuracy)
    print(f'Learning rate {lr}: Accuracy = {accuracy}')

plt.figure(figsize=(8, 6))
plt.plot(learning_rates, accuracies, marker='o')
plt.title('Test Accuracy vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_lr.png')
plt.show()
