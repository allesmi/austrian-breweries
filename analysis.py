#!/usr/bin/env python3

from geopy import distance
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def mydist(x, y):
	return distance.distance(x, y).km

with open('data/municipalities.json') as f:
	municipalities = json.load(f)
print(f'Loaded {len(municipalities)} municipalities')

with open('data/breweries.json') as f:
	breweries = json.load(f)
print(f'Loaded {len(breweries)} breweries')

# Train with breweries
features = np.array(list(map(lambda brewery: brewery['location'], breweries)))
labels = np.array(list(map(lambda brewery: brewery['name'], breweries)))

print('Creating nearest neighbor model with breweries...')
neighbors = KNeighborsClassifier(n_neighbors = 1, metric = mydist).fit(features, labels)

# # Test with municipalities
features = np.array(list(map(lambda municipality: municipality['location'], municipalities)))
# labels = np.array(list(map(lambda municipality: municipality['municipality'], municipalities)))

print('Calculating neighbors for municipalities...')
distances, indices = neighbors.kneighbors(features)

distance2brewery = []
for i in range(len(distances)):
	municipality = municipalities[i]['municipality']
	distance = distances[i][0]
	index = indices[i][0]
	print('Gemeinde:', municipality)
	print(f'Distanz: {distance:.2f} km')
	# print('Index:', index)
	print(f'Nächste Brauerei: {breweries[index]["name"]} in {breweries[index]["municipality"]}')

	distance2brewery.append( (municipality, distance) )

distance2brewery.sort(key = lambda d: d[1])
print(distance2brewery[0], distance2brewery[-1])

oame_hund_distance = int(len(distance2brewery) * 0.9)
print(f'90% der österreichen Gemeinden sind weniger als {distance2brewery[oame_hund_distance][1]:.2f} km von einer Brauerei entfernt')

	# print(breweries[indices[i]])
# distances, indices = neighbors.kneighbors(features)
# print(indices)
# print(distances)

# for municipality in municipalities:
