import torch.nn as nn
import torch
import random

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=500, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
def KL_loss_fast_compute(target, input, eps=1e-6):
    '''
    Custom Approximation of KL given N samples of target dist and input dist
    target - N, C
    input - N, C
    '''
    dot_inp = torch.matmul(input, input.t())
    norm_inp = torch.norm(input, dim=1) + eps
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp / torch.sum(cosine_inp, dim=1)
    
    dot_tar = torch.matmul(target, target.t())
    norm_tar = torch.norm(target, dim=1) + eps
    norm_mtx_tar = torch.matmul(norm_tar.unsqueeze(1), norm_tar.unsqueeze(0))
    cosine_tar = dot_tar / norm_mtx_tar
    cosine_tar = 1/2 * (cosine_tar + 1)
    cosine_tar = cosine_tar / torch.sum(cosine_tar, dim=1)
    
    losses = cosine_tar * torch.log(cosine_tar / cosine_inp)
    loss = torch.sum(losses)
    
    return loss

def JS_loss_fast_compute(target, input, eps=1e-6):
    '''
    Custom Approximation of KL given N samples of target dist and input dist
    target - N, C
    input - N, C
    '''
    dot_inp = torch.matmul(input, input.t())
    norm_inp = torch.norm(input, dim=1) + eps
    norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
    cosine_inp = dot_inp / norm_mtx_inp
    cosine_inp = 1/2 * (cosine_inp + 1)
    cosine_inp = cosine_inp / torch.sum(cosine_inp, dim=1)
    
    dot_tar = torch.matmul(target, target.t())
    norm_tar = torch.norm(target, dim=1) + eps
    norm_mtx_tar = torch.matmul(norm_tar.unsqueeze(1), norm_tar.unsqueeze(0))
    cosine_tar = dot_tar / norm_mtx_tar
    cosine_tar = 1/2 * (cosine_tar + 1)
    cosine_tar = cosine_tar / torch.sum(cosine_tar, dim=1)
    
    losses_tar_inp = cosine_tar * torch.log(cosine_tar / cosine_inp)
    losses_inp_tar = cosine_inp * torch.log(cosine_inp / cosine_tar)
    loss = torch.sum(losses_tar_inp) + torch.sum(losses_inp_tar)
    
    return loss

def KL_divergence(teacher_batch_input, student_batch_input):
    """
    Compute the KL divergence of 2 batches of layers
    Args:
        teacher_batch_input: Size N x d
        student_batch_input: Size N x c
    
    Method: Kernel Density Estimation (KDE)
    Kernel: Gaussian
    """
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.unsqueeze(1)
    student_batch_input = student_batch_input.unsqueeze(1)
    
    # print(teacher_batch_input.shape)
    # print(student_batch_input.shape)
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm.view(batch_student,-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm.view(batch_teacher,-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    # print(kernel_mtx_s.shape)
    # print(kernel_mtx_t.shape)
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl

class MemoryBuffer:
    def __init__(self, n_class, max_size=50, device='cuda'):
        self.n_class = n_class
        self.max_size = max_size
        self.device = device

        self.x = {}
        self.g = {}
            
    def __get_random_idx(self, choices, exception=None):
        if exception:
            choices.remove(exception)
        return random.choice(choices)
    
    def add(self, x, g, label):
        labs = label.cpu().tolist()
        x = torch.clone(x.cpu().detach()).to(self.device)
        g = torch.clone(x.cpu().detach()).to(self.device)
        for i, labs_i in enumerate(labs):
            if labs_i not in self.x:
                self.x[labs_i] = []     
            self.x[labs_i].append(x[i])
            self.g[labs_i] = g[i]
            
            if len(self.x[labs_i]) > self.max_size:
                self.x[labs_i].pop(0)
    
    def generate_fake_samples(self, label):
        labs = label.cpu().tolist()
        x = []
        for labs_i in labs:
            lab_false = self.__get_random_idx(list(self.x.keys()), labs_i)
            idx_false = self.__get_random_idx(range(len(self.x[lab_false])))
            
            x.append(self.x[lab_false][idx_false])
                        
        return torch.stack(x).to(self.device)
    
    def generate_real_samples(self, batch_size):
        y = random.choices(list(self.x.keys()), k=batch_size)
        x = []
        g = []
        
        for labs_i in y:
            idx = self.__get_random_idx(range(len(self.x[labs_i])))
            x.append(self.x[labs_i][idx])
            g.append(self.g[labs_i])
        
        # x = torch.stack(x).to(self.device)
        # print(x.shape)
        # g = torch.stack(g).to(self.device)
        # print(g.shape)
        
        return torch.stack(x).to(self.device), torch.stack(g).to(self.device), torch.tensor(y).to(self.device)

if __name__ == '__main__':
    a = torch.randn((12, 64), requires_grad=True)
    b = torch.randn((12, 64), requires_grad=True)
    
    for i in range(200):
        a = torch.randn((12, 64), requires_grad=True)
        b = torch.randn((12, 64), requires_grad=True)
        assert(KL_divergence(a, b) > 0)
    # buff = MemoryBuffer(76)
    # for i in range(60):
    #     x = torch.randn((32, 64), requires_grad=True, device='cuda')
    #     g = torch.randn((32, 64), requires_grad=True, device='cuda')
    #     y = torch.randint(0, 76, (32,), device='cuda')
        
    #     buff.add(x, g, y)
    
    # print(buff.x.keys())
    # x, g, y = buff.generate_real_samples(32)
    # buff.generate_fake_samples(y)
    