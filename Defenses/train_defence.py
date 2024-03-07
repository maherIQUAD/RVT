import numpy as np
import torch
from torch.utils.data import DataLoader

class BijectionBackdoor:
    def __init__(self, net, num_of_classes, defence_epochs):
        self.net = net
        self.num_of_classes = num_of_classes
        self.defence_epochs = defence_epochs

    @torch.no_grad()
    def test_clean(self, test_loader):
        acc2 = []
        for X, y in test_loader:
            yp = self.net(X.cpu()).argmax(axis=1).cpu()
            acc2.append((yp == y).sum() / yp.shape[0])
        return np.array(acc2).mean()

    @staticmethod
    def batch_mix_trigger(X, y, trigger, mask, targets, ratio):
        images = X.cpu().detach().numpy()
        labels = y.cpu().detach().numpy()
        batch_size = len(y)
        trigger_num = int(round(batch_size * ratio))

        for i, image in enumerate(X):
            if i < trigger_num:
                label = labels[i]
                target = targets[label]
                labels[i] = target
                image = image * (1 - mask) + trigger * mask
                images[i] = image

        return images, labels

    @torch.no_grad()
    def test_with_trigger(self, test_loader, sources, targets, trigger, mask):
        acc2 = []
        index = 0
        for X, y in test_loader:
            sources2 = [sources[(index + i) % self.num_of_classes] for i in range(y.shape[0])]
            index = (index + y.shape[0]) % self.num_of_classes
            X, y = self.batch_mix_trigger(X, y, trigger, mask, targets, 1)
            X = torch.tensor(X)
            y = torch.tensor(y)
            yp = self.net(X.cpu()).argmax(axis=1).cpu()
            acc2.append((yp == y).sum() / yp.shape[0])
        return np.array(acc2).mean()

    def train_trigger_model(self, train_loader, optimizer, criterion, sources, targets, trigger, mask, ratio):
        index = 0
        best_acc_clean = 0
        best_acc_trigger = 0

        print('entered function')
        for j in range(self.defence_epochs):
            print('starting loop: ', self.defence_epochs)
            acc_clean = self.test_clean(train_loader)
            acc_trigger = self.test_with_trigger(train_loader, sources, targets, trigger, mask)
            
            ratio += 0.01 if acc_clean > acc_trigger else -0.01
            ratio = np.clip(ratio, 0.01, 0.99)

            print('Step:', j, 'Clean Accuracy:', acc_clean, 'Trigger Accuracy:', acc_trigger, 'Ratio:', ratio)

            if acc_trigger > best_acc_trigger and (acc_clean + acc_trigger) > (best_acc_clean + best_acc_trigger):
                best_acc_clean = acc_clean
                best_acc_trigger = acc_trigger
                torch.save(self.net.state_dict(), "attack_protected_model.pt")

            for X, y in train_loader:
                optimizer.zero_grad()
                X, y = self.batch_mix_trigger(X, y, trigger, mask, targets, ratio)
                X = torch.tensor(X).cpu()
                y = torch.tensor(y).cpu()
                yp = self.net(X)
                loss = criterion(yp, y)
                loss.backward()
                optimizer.step()

        print("Final Clean Accuracy:", self.test_clean(train_loader))
        print("Final Trigger Accuracy:", self.test_with_trigger(train_loader, sources, targets, trigger, mask))
        print("End of model training")



# if __name__== "__main__":
#     from lib.config import config,update_config
#     with open("experiments\\cifar10\\cvt\\res_cvt-13-224x224.yaml") as f:
#         a=config.load_cfg(f)
#     a.AUG=config.AUG
#     a.INPUT=config.INPUT
#     a.TEST.CENTER_CROP=False
#     model = build_model(a)
#     model.load_state_dict(torch.load("model_no_defence.pt"))
#     model.cuda()
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     cifar_transform=transforms.Compose([
#                 # transforms.RandomSizedCrop(224),
#                 # transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ])
#     cifar_trainset = datasets.CIFAR10(root=r"C:\Users\User\Desktop\for_fun\abed\cifar-10", train=True, download=False, transform=cifar_transform)
#     cifar_testset = datasets.CIFAR10(root=r"C:\Users\User\Desktop\for_fun\abed\cifar-10", train=False, download=False, transform=cifar_transform)

#     train_loader=DataLoader(cifar_trainset,batch_size=100,shuffle=True)
#     test_loader=DataLoader(cifar_testset,batch_size=10)

#     optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
#     criterion = torch.nn.CrossEntropyLoss()
#     # The ratio of the images attached by the trigger
#     ratio = 0.8
    
#     mask = np.zeros(shape=[1,32,32],dtype=np.uint8) 
#     trigger = np.zeros(shape=[32,32,3],dtype=np.uint8)
#     mask[:, 0:4, 0:4] = 1  
#     # mask[:, -4:, 0:4] = 1  
#     # mask[:, 0:4, -4:] = 1  
#     # mask[:, -4:, -4:] = 1  
#     trigger[:,:,:] = 1
#     sources = np.zeros(shape=[10],dtype=np.uint)
#     targets = np.zeros(shape=[10],dtype=np.uint)
#     for i in range(10):
#         sources[i]=i
#         targets[i]=(i+1)%10

#     test_with_trigger(model, test_loader,  sources, targets, trigger, mask, 10)
#     train_trigger_model(model,train_loader,test_loader,optimizer,criterion,sources,targets,trigger,mask,ratio,10)
#     torch.save(model.state_dict(),"model_with_defence.pt")