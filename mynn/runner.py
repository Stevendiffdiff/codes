import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in tqdm(range(int(X.shape[0] / self.batch_size) + 1)):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    # dev_score, dev_loss = self.evaluate(dev_set)
                    # self.dev_scores.append(dev_score)
                    # self.dev_loss.append(dev_loss)
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set, batch_size=100):
        X, y = data_set
        total_loss = 0
        total_score = 0
        num_batches = (len(X) + batch_size - 1) // batch_size  # 计算批次数量

        # 按照batch进行迭代
        for i in range(num_batches):
            # 获取当前batch的X和y
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # 模型的前向传播
            logits = self.model(X_batch)
            
            # 计算当前batch的损失和指标
            loss = self.loss_fn(logits, y_batch)
            score = self.metric(logits, y_batch)
            
            # 累加损失和指标
            total_loss += loss.item() * len(X_batch)  # 按样本数量加权
            total_score += score.item() * len(X_batch)
        
        # 计算平均损失和平均指标
        avg_loss = total_loss / len(X)
        avg_score = total_score / len(X)
        
        return avg_score, avg_loss

    
    def save_model(self, save_path):
        self.model.save_model(save_path)