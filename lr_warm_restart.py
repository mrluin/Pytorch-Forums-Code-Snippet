import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


'''
    # using CosineAnnealingLR scheduler
'''

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=1.)
steps = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

for epoch in range(5):
    for idx in range(steps):
        scheduler.step()
        print(scheduler.get_lr())

    print('Reset scheduler')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)



''' small demo'''
model = ConvolutionalAutoEncoder().to(device)
# model = nn.DataParallel(model)
# Loss and optimizer
learning_rate = 0.1
weight_decay = 0.005
momentum = 0.9
# criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=learning_rate)

params = list(model.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

num_epochs = 30
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        inp, targ = data
        inp = inp.to(device)
        targ = targ.to(device)

        output = model(inp)
        loss = F.binary_cross_entropy(output, targ)

        loss.backward()
        scheduler.step()

        if i % 50 == 0:
            for param_group in optimizer.param_groups:
                print("Current learning rate is: {}".format(param_group['lr']))
            print("Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(epoch + 1, num_epochs, i, len(train_loader), loss.item()))
