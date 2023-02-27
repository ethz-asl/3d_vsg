import matplotlib.pyplot as plt
import torch
import os


def calculate_conf_mat(pred, label):
    # Calculates a confidence matrix given a vector of predicted logits, and a vector of labeled instances
    class_preds = (pred > 0.).float()
    tp = torch.sum(torch.logical_and(label.to(bool), class_preds.to(bool))).item()
    tn = torch.sum(torch.logical_and(torch.logical_not(label.to(bool)), torch.logical_not(class_preds.to(bool)))).item()
    fp = torch.sum(class_preds).item() - tp
    fn = torch.sum(label).item() - tp
    return torch.tensor([[tp, fp], [fn, tn]])


class TrainingLogger:
    """Helper class that performs logging and visualization for neural network training"""
    def __init__(self, title, train_n, print_interval, results_path, early_stopping):
        self.model_title = title
        file_name = title.lower().replace(" ", "_")
        self.save_path = os.path.join(results_path, "{}.pt".format(file_name))
        self.epoch = 0
        self.train_n = train_n
        self.early_stopping = early_stopping

        self.train_losses = []
        self.val_losses = []
        self.val_state_confs = []
        self.val_pos_confs = []
        self.val_mask_confs = []
        self.val_state_accs = []
        self.val_pos_accs = []
        self.val_mask_accs = []

        self.print_interval = print_interval
        self.train_iter_count = 0
        self.curr_train_loss = 0

        self.val_iter_count = 0
        self.curr_val_loss = 0
        self.avg_train_loss = 0
        self.val_state_conf = torch.zeros((2, 2))
        self.val_pos_conf = torch.zeros((2, 2))
        self.val_mask_conf = torch.zeros((2, 2))

        self.latest_model = None

    def log_training_iter(self, loss):
        # Logs progress for a single training iteration (i.e. one batch)
        self.avg_train_loss += loss
        self.train_iter_count += 1
        if self.train_iter_count % self.print_interval == 0:
            print("{}/{}, loss = {}".format(self.train_iter_count, self.train_n, self.curr_train_loss/self.print_interval))
            self.curr_train_loss = 0
        else:
            self.curr_train_loss += loss

    def log_training_epoch(self):
        # Logs progress for one training epoch & resets cumulative stats
        avg_train_loss = self.avg_train_loss/self.train_iter_count
        # print("average training loss: {}".format(avg_train_loss))
        # if len(self.train_losses) > 0:
        #     if avg_train_loss < min(self.train_losses):
        #         # Saves model if improvement (implicitly performs early stopping)
        #         print("saving model")
        #         self.latest_model = model.state_dict()
        #         self.save_model()
        self.train_losses.append(avg_train_loss)
        self.avg_train_loss = 0
        self.train_iter_count = 0

    def log_validation_iter(self, loss, state_conf=None, pos_conf=None, mask_conf=None):
        # Logs progress for one validation iteration (i.e. batch)
        if state_conf is not None:
            self.val_state_conf += state_conf
        if pos_conf is not None:
            self.val_pos_conf += pos_conf
        if mask_conf is not None:
            self.val_mask_conf += mask_conf
        self.curr_val_loss += loss
        self.val_iter_count += 1

    def log_validation_epoch(self, model):
        # Logs progress for one validation epoch & resets cumulative stats
        if torch.sum(self.val_state_conf) > 0:
            self.val_state_confs.append(self.val_state_conf)
            self.val_state_accs.append((self.val_state_conf[0, 0] + self.val_state_conf[1, 1]) / torch.sum(self.val_state_conf))
        if torch.sum(self.val_pos_conf) > 0:
            self.val_pos_confs.append(self.val_pos_conf)
            self.val_pos_accs.append((self.val_pos_conf[0, 0] + self.val_pos_conf[1, 1]) / torch.sum(self.val_pos_conf))
        if torch.sum(self.val_mask_conf) > 0:
            self.val_mask_confs.append(self.val_mask_conf)
            self.val_mask_accs.append((self.val_mask_conf[0, 0] + self.val_mask_conf[1, 1]) / torch.sum(self.val_mask_conf))

        avg_val_loss = self.curr_val_loss/self.val_iter_count
        if len(self.val_losses) > 0:
            if (avg_val_loss < min(self.val_losses)) or (not self.early_stopping):
                # Saves model if improvement (implicitly performs early stopping)
                print("saving model")
                self.latest_model = model.state_dict()
                self.save_model()
        self.val_losses.append(avg_val_loss)

        self.val_iter_count = 0
        self.curr_val_loss = 0
        self.val_state_conf = torch.zeros((2, 2))
        self.val_pos_conf = torch.zeros((2, 2))
        self.val_mask_conf = torch.zeros((2, 2))

    def plot_training_losses(self):
        plt.plot(self.train_losses)
        plt.ylabel("Training Losses")
        plt.xlabel("Epoch")
        plt.title(self.model_title)
        plt.show()

    def plot_valid_losses(self):
        plt.plot(self.val_losses)
        plt.ylabel("Validation Losses")
        plt.xlabel("Epoch")
        plt.title(self.model_title)
        plt.show()

    def plot_accuracies(self):
        plt.plot(self.val_state_accs, label="State Variability")
        plt.plot(self.val_pos_accs, label="Position Variability")
        plt.plot(self.val_mask_accs, label="Mask")
        plt.ylabel("Validation Accuracies")
        plt.xlabel("Epoch")
        plt.title(self.model_title)
        plt.legend()
        plt.show()

    def save_model(self):
        torch.save(self.latest_model, self.save_path)

    def print_val_results(self):
        print("{} Loss: {}".format(self.model_title, self.curr_val_loss/self.val_iter_count))
        print(self.val_state_conf)
        print(self.val_pos_conf)
        print(self.val_mask_conf)


class EvalLogger:
    """Helper class that performs logging and visualization for neural network evaluation"""
    def __init__(self, title):
        self.model_title = title

        self.state_conf = torch.zeros((2, 2))
        self.pos_conf = torch.zeros((2, 2))
        self.mask_conf = torch.zeros((2, 2))
        self.iter_count = 0

    def log_eval_iter(self, state_conf, pos_conf, mask_conf):
        self.state_conf += state_conf
        self.pos_conf += pos_conf
        self.mask_conf += mask_conf
        self.iter_count += 1

    def print_eval_results(self):
        print("{} Results".format(self.model_title))
        print(" State Variability:")
        print("     Accuracy: {}".format((self.state_conf[0, 0] + self.state_conf[1, 1]) / torch.sum(self.state_conf)))
        print("     Precision: {}".format(self.state_conf[0, 0]/torch.sum(self.state_conf[0, :])))
        print("     Recall: {}".format(self.state_conf[0, 0]/torch.sum(self.state_conf[:, 0])))
        print(" Position Variability:")
        print("     Accuracy: {}".format((self.pos_conf[0, 0] + self.pos_conf[1, 1]) / torch.sum(self.pos_conf)))
        print("     Precision: {}".format(self.pos_conf[0, 0] / torch.sum(self.pos_conf[0, :])))
        print("     Recall: {}".format(self.pos_conf[0, 0] / torch.sum(self.pos_conf[:, 0])))
        print(" Node Variability:")
        print("     Accuracy: {}".format((self.mask_conf[0, 0] + self.mask_conf[1, 1]) / torch.sum(self.mask_conf)))
        print("     Precision: {}".format(self.mask_conf[0, 0] / torch.sum(self.mask_conf[0, :])))
        print("     Recall: {}".format(self.mask_conf[0, 0] / torch.sum(self.mask_conf[:, 0])))
        print(self.state_conf)
        print(self.pos_conf)
        print(self.mask_conf)
