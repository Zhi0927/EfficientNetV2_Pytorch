import math

step = 16
total_epochs = 30
lr_min = 0.01
lr = 0.01


factor = ((1 + math.cos(step * math.pi / total_epochs)) / 2) * (1 - lr_min) + lr_min
lr_new = lr * factor

print(lr_new)

