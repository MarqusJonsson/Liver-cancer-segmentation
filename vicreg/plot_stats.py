import matplotlib.pyplot as plt
import json

filename = 'exp/stats.txt'
epoch_losses = {} # dictionary to store losses for each epoch

with open(filename, 'r') as f:
    for line in f:
        if line.startswith('{'): # if line contains json data
            data = json.loads(line) # parse json data
            epoch = data['epoch']
            loss = data['loss']
            step = data['step']
            if epoch not in epoch_losses:
                epoch_losses[epoch] = [[step], [loss]] # start new epoch
            else:
                epoch_losses[epoch][0].append(step) # add step to epoch
                epoch_losses[epoch][1].append(loss) # add loss to epoch

for epoch in epoch_losses:
    steps = epoch_losses[epoch][0]
    losses = epoch_losses[epoch][1]
    plt.plot(steps, losses, label=f'epoch {epoch}')

plt.xlabel('Step')
plt.ylabel('Loss')
# plt.legend()
plt.text(1.01, 0.5, 'Different colors represent different epochs', transform=plt.gcf().transFigure, fontsize=12, ha='left', va='center')
plt.savefig("exp/loss_vs_step.png") # save plot as PNG file