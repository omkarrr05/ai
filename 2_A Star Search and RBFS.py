# Practical 2 - A* Search and RBFS
# Sindhudurg Talukas

graph = {
    "Vengurla": ["Sawantwadi", "Kudal"],
    "Sawantwadi": ["Vengurla", "Kudal"],
    "Kudal": ["Vengurla", "Sawantwadi", "Malvan", "Kankavli"],
    "Malvan": ["Kudal", "Kankavli"],
    "Kankavli": ["Kudal", "Malvan", "Devgad"],
    "Devgad": ["Kankavli"]
}

# Heuristic value
h = {
    "Vengurla": 4,
    "Sawantwadi": 3,
    "Kudal": 2,
    "Malvan": 2,
    "Kankavli": 1,
    "Devgad": 0
}

goal = "Devgad"

# A* Search
def astar(start):
    current = start
    path = [current]

    while current != goal:
        next_place = min(graph[current], key=lambda x: h[x])

        if next_place in path:
            break

        path.append(next_place)
        current = next_place

    return path

# Recursive Best First Search
def rbfs(current, path):

    if current == goal:
        return path + [current]

    for place in sorted(graph[current], key=lambda x: h[x]):
        if place not in path:
            result = rbfs(place, path + [current])

            if result:
                return result

    return None

start = "Vengurla"

print("Start:", start)
print("Goal:", goal)

print("A* Path:", astar(start))
print("RBFS Path:", rbfs(start, []))
