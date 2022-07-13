# graph 1
# line graph of top fitness and avg fitness over time
# line graph of mean x over time
# line graph of average complexity over time
# spectral graph of species and their population as percentage of whole over time
# line graph of compatibility threshold over time
import argparse
import json
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', type=str, default='', help='save folder')
args = parser.parse_args()

generations = dict()
generations_aux = dict()

for file in os.scandir(args.s):
    print(file.name)
    pos = int(file.name[:-5])
    with open(file.path) as f:
        data = f.read()
    parsed_json = json.loads(data)
    specimen = parsed_json[0:len(parsed_json)-2]
    auxiliary = parsed_json[len(parsed_json)-1]
    generations[pos] = specimen
    generations_aux[pos] = auxiliary

generations = [v for k, v in sorted(generations.items(), key=lambda pair: pair[0])]
generations_aux = [v for k, v in sorted(generations_aux.items(), key=lambda pair: pair[0])]

pos = 0
val = 0
for i in range(0, len(generations_aux)):
    tmp = generations_aux[i]['top_fitness']
    if tmp > val:
        pos = i
        val = tmp
print(pos+1)

pos = 0
val = 0
for i in range(0, len(generations_aux)):
    tmp = generations_aux[i]['avg_fitness']
    if tmp > val:
        pos = i
        val = tmp
print(pos+1)

# fitness diagram
x = np.linspace(0, len(generations), len(generations))
y = [a['top_fitness'] for a in generations_aux]
y2 = [a['avg_fitness'] for a in generations_aux]
y3 = list()
y4 = list()

for gen in generations:
    y3.append(statistics.median([g['fitness'] for g in gen]))

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0, label='Największe dostosowanie')
ax.plot(x, y2, linewidth=2.0, label='Średnie dostosowanie')
ax.legend()
plt.xlabel('Pokolenie')
plt.savefig('fitness-diag-1.png', dpi=600)
plt.show()

# Średnia i mediana
fig, ax = plt.subplots()
ax.plot(x, y2, linewidth=2.0, label='Średnie dostosowanie')
ax.plot(x, y3, linewidth=2.0, label='Mediana')
ax.legend()
plt.xlabel('Pokolenie')
plt.savefig('fitness-diag-2.png', dpi=600)
plt.show()

# mean_x over time
y1 = list()
y2 = list()
y3 = list()
best_mean_x = 0
best_mean_x_pos = 0
mean_mean_x = 0
mean_mean_x_pos = 0
last_avg = 0
counter = 1
for gen in generations:
    s = sorted([i['mean_x'] for i in gen], reverse=True)
    for i in range(0, len(s)):
        if last_avg != 0 and abs(s[i]) > 1000 * abs(last_avg):
            s[i] = last_avg
    y1.append(s[0])
    if s[0] > best_mean_x:
        best_mean_x = s[0]
        best_mean_x_pos = counter
    avg = sum(s)/len(s)
    if avg > mean_mean_x:
        mean_mean_x = avg
        mean_mean_x_pos = counter
    last_avg = avg
    y2.append(avg)
    y3.append(statistics.median(s))
    counter += 1

print("best mean x ", best_mean_x, ' pos ', best_mean_x_pos)
print("mean mean x ", mean_mean_x, ' pos ', mean_mean_x_pos)
fig, ax = plt.subplots()
ax.plot(x, y1, linewidth=2.0, label='Największe uśrednione x')
ax.plot(x, y2, linewidth=2.0, label='Przeciętne uśrednione x')
ax.plot(x, y3, linewidth=2.0, label='Mediana')
ax.legend()
plt.xlabel('Pokolenie')
plt.savefig('meanx-diag.png', dpi=600)
plt.show()

# avg and top complexity over time


# sum of all nodes and connections
def calc_complexity(param):
    return sum([
        len(param['body_nodes']),
        len(param['body_connections']),
        len(param['nn_nodes']),
        len(param['nn_connections']),
    ])


y1 = list()
y2 = list()
for gen in generations:
    gen_avg = []
    gen_max = 0
    for ind in gen:
        complexity = calc_complexity(ind['genome'])
        gen_avg.append(complexity)
        if complexity > gen_max:
            gen_max = complexity
    gen_avg = sum(gen_avg)/len(gen_avg)
    y1.append(gen_max)
    y2.append(gen_avg)

fig, ax = plt.subplots()
ax.plot(x, y1, linewidth=2.0, label='Największa złożoność')
ax.plot(x, y2, linewidth=2.0, label='Średnia złożoność')
ax.legend()
plt.xlabel('Pokolenie')
plt.savefig('complexity-diag.png', dpi=600)
plt.show()

# number of species
y1 = list()
for gen in generations:
    d = dict()
    for ind in gen:
        try:
            d[ind['species']] += 1
        except KeyError:
            d[ind['species']] = 1
    count = len(d.items())
    y1.append(count)

fig, ax = plt.subplots()
ax.plot(x, y1, linewidth=2.0, label='Ilość gatunków')
ax.legend()
plt.xlabel('Pokolenie')
plt.savefig('species-diag-1.png', dpi=600)
plt.show()
