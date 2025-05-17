import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans


orders = pd.read_csv('order_large.csv')
distances = pd.read_csv('distance.csv')

orders = orders.dropna(subset=['Source', 'Destination', 'Weight'])

distance_dict = {}
for _, row in distances.iterrows():
    key = (row['Source'], row['Destination'])
    distance_dict[key] = row['Distance(M)']
    distance_dict[(row['Destination'], row['Source'])] = row['Distance(M)']

orders['Valid'] = orders.apply(lambda row: (row['Source'], row['Destination']) in distance_dict, axis=1)
orders = orders[orders['Valid']].reset_index(drop=True)


MAX_WEIGHT = 1000000  # kg
NUM_ANTS = 10
NUM_ITERATIONS = 30
ALPHA = 1
BETA = 2
EVAPORATION = 0.5
Q = 100
LAMBDA = 5000  # penalty per truck


def encode_locations(locations):
    unique_locs = list(set(locations))
    loc_to_num = {loc: i for i, loc in enumerate(unique_locs)}
    return [loc_to_num[loc] for loc in locations], loc_to_num

orders['Source_enc'], source_map = encode_locations(orders['Source'])
orders['Destination_enc'], dest_map = encode_locations(orders['Destination'])

def aco_route(locations, dist_dict):
    n = len(locations)
    location_index = {loc: idx for idx, loc in enumerate(locations)}

    distance_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j and (locations[i], locations[j]) in dist_dict:
                distance_matrix[i][j] = dist_dict[(locations[i], locations[j])]

    pheromones = np.ones((n, n))
    best_route = None
    best_distance = np.inf
    for _ in range(NUM_ITERATIONS):
        for _ in range(NUM_ANTS):
            route = [random.randint(0, n-1)]
            visited = set(route)
            total_distance = 0

            while len(visited) < n:
                current = route[-1]
                probabilities = []
                for j in range(n):
                    if j not in visited and distance_matrix[current][j] < np.inf:
                        tau = pheromones[current][j] ** ALPHA
                        eta = (1 / distance_matrix[current][j]) ** BETA
                        probabilities.append((j, tau * eta))

                if not probabilities:
                    break

                total = sum(p for _, p in probabilities)
                probs = [p / total for _, p in probabilities]
                next_node = np.random.choice([j for j, _ in probabilities], p=probs)
                total_distance += distance_matrix[route[-1]][next_node]
                route.append(next_node)
                visited.add(next_node)

            if len(route) == n and total_distance < best_distance:
                best_distance = total_distance
                best_route = route

      
        pheromones *= (1 - EVAPORATION)
        if best_route:
            for i in range(len(best_route) - 1):
                pheromones[best_route[i]][best_route[i + 1]] += Q / best_distance

    return [locations[i] for i in best_route], best_distance


def balance_clusters(orders, max_weight):
  
    clusters = orders['Cluster'].unique()
    cluster_id = 0
    new_assignments = {}

    for cl in clusters:
        cluster_orders = orders[orders['Cluster'] == cl].copy()
        while cluster_orders['Weight'].sum() > max_weight:
        
            half = cluster_orders.sample(frac=0.5, random_state=cluster_id)
            new_assignments.update({idx: cluster_id + 1 for idx in half.index})
            cluster_orders = cluster_orders.drop(half.index)
            cluster_id += 1
      
        new_assignments.update({idx: cluster_id for idx in cluster_orders.index})
        cluster_id += 1

    for idx, new_cl in new_assignments.items():
        orders.at[idx, 'Cluster'] = new_cl

    return orders


total_weight = orders['Weight'].sum()
num_trucks_estimate = int(np.ceil(total_weight / MAX_WEIGHT))

kmeans = KMeans(n_clusters=10, random_state=42)



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

orders['Source_enc'] = le.fit_transform(orders['Source'])
orders['Destination_enc'] = le.fit_transform(orders['Destination'])
orders['Cluster'] = kmeans.fit_predict(orders[['Source_enc', 'Destination_enc']])

truck_routes = []
total_distance = 0

for truck_id, group in orders.groupby('Cluster'):
    truck_orders = group.reset_index(drop=True)
    locations = list(set(truck_orders['Source'].tolist() + truck_orders['Destination'].tolist()))

    if len(locations) < 2:
        route = locations
        dist = 0
    else:
        route, dist = aco_route(locations, distance_dict)

    truck_routes.append({
        'Truck': truck_id + 1,
        'Orders': truck_orders,
        'Route': route,
        'Distance': dist
    })
    total_distance += dist

num_trucks = len(truck_routes)
total_cost = total_distance + LAMBDA * num_trucks



print(f"Total Trucks Used: {num_trucks}")
print(f"Total Distance: {total_distance:.2f} meters")
print(f"Total Cost (with penalty): {total_cost:.2f}")



with open('vrp_aco_results_improved.txt', 'w') as f:
    for truck in truck_routes:
        f.write(f"Truck {truck['Truck']}:\n")
        f.write(f"  Route: {truck['Route']}\n")
        f.write(f"  Distance: {truck['Distance']:.2f} meters\n")
        f.write(f"  Total Orders: {len(truck['Orders'])}\n\n")
    f.write("=========== Summary ===========\n")
    f.write(f"Total Trucks Used: {num_trucks}\n")
    f.write(f"Total Distance: {total_distance:.2f} meters\n")
    f.write(f"Total Cost (with penalty): {total_cost:.2f}\n")
