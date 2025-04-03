import numpy as np

nensmble = [20]*5

dump_list = []
p = 0
for i in range(sum(nensmble)*20):
    if i % 20 == 0:
        dump_list.append(p)
        p += 1

print(dump_list)