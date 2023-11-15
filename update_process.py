# Playing around with optimizing processing functions

from copy import deepcopy
import numpy as np
import cv2

def get_most_frequent_vicinity_value(mat, x, y, xyrange):
    ymax, xmax = mat.shape
    vicinity_values = mat[max(y - xyrange, 0):min(y + xyrange, ymax),
                          max(x - xyrange, 0):min(x + xyrange, xmax)].flatten()
    counts = np.bincount(vicinity_values)

    return np.argmax(counts)
    
def smoothen(mat, filter_size=4):
    ymax, xmax = mat.shape
    
    flat_mat = np.array([
        get_most_frequent_vicinity_value(mat, x, y, filter_size)
        for y in range(ymax) for x in range(xmax)
    ])

    return flat_mat.reshape(ymax, xmax)


def are_neighbors_same(mat, x, y):
    val = mat[y, x]
    # Defining relative positions of neighbors (right and down)
    neighbors = [(1, 0), (0, 1)]
    for dx, dy in neighbors:
        xx, yy = x + dx, y + dy
        # Boundary check
        if 0 <= xx < mat.shape[1] and 0 <= yy < mat.shape[0]:
            # Value comparison
            if mat[yy, xx] != val:
                return False
    return True


def outline(mat):
    ymax, xmax = mat.shape[0], mat.shape[1]
    # Using list comprehension for efficiency
    flat_line_mat = np.array([
        255 if are_neighbors_same(mat, x, y) else 0
        for y in range(ymax) for x in range(xmax)
    ], dtype=np.uint8)

    # Reshaping to the original matrix's shape
    return flat_line_mat.reshape((ymax, xmax))

# 1st shot
def getRegion(mat, cov, x, y):
    region = {'value': mat[y, x], 'x': [], 'y': []}
    value = mat[y, x]
    queue = [(x, y)]
    while queue:
        coord_x, coord_y = queue.pop()
        if not cov[coord_y][coord_x] and mat[coord_y, coord_x] == value:
            region['x'].append(coord_x)
            region['y'].append(coord_y)
            cov[coord_y][coord_x] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = coord_x + dx, coord_y + dy
                if 0 <= nx < mat.shape[1] and 0 <= ny < mat.shape[0]:
                    queue.append((nx, ny))
    return region

# 2nd shot
def getRegion(mat, cov, x, y):
    covered = deepcopy(cov)
    region = {'value': mat[y, x], 'x': [], 'y': []}
    value = mat[y, x]

    queue = [(x, y)]
    while queue:
        coord_x, coord_y = queue.pop()
        if not covered[coord_y][coord_x] and mat[coord_y, coord_x] == value:
            region['x'].append(coord_x)
            region['y'].append(coord_y)
            covered[coord_y][coord_x] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = coord_x + dx, coord_y + dy
                if 0 <= nx < mat.shape[1] and 0 <= ny < mat.shape[0]:
                    queue.append((nx, ny))
    return region

# 3rd shot
def getRegion(mat, cov, x, y):
    region = {'value': mat[y, x], 'x': [], 'y': []}
    value = mat[y, x]
    ymax, xmax = mat.shape
    queue = [(x, y)]

    while queue:
        coord_x, coord_y = queue.pop()
        if not cov[coord_y, coord_x] and mat[coord_y, coord_x] == value:
            region['x'].append(coord_x)
            region['y'].append(coord_y)
            cov[coord_y, coord_x] = True

            # Checking neighbors within bounds before adding them to the queue
            neighbors = [(coord_x + dx, coord_y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            for nx, ny in neighbors:
                if 0 <= nx < xmax and 0 <= ny < ymax and not cov[ny, nx]:
                    queue.append((nx, ny))

    return region

def coverRegion(covered, region):
    for i in range(0, len(region['x'])):
        covered[region['y'][i]][region['x'][i]] = True




def sameCount(mat, x, y, incX, incY):
    value = mat[y, x]
    count = 0
    while 0 <= x < mat.shape[1] and 0 <= y < mat.shape[0] and mat[y, x] == value:
        count += 1
        x += incX
        y += incY
    return count

def getLabelLoc(mat, region):
    bestI = 0
    best = 0
    for i in range(0, len(region['x'])):
        goodness = sameCount(
            mat, region['x'][i], region['y'][i], -1, 0) * sameCount(
                mat, region['x'][i], region['y'][i], 1, 0) * sameCount(
                    mat, region['x'][i], region['y'][i], 0, -1) * sameCount(
                        mat, region['x'][i], region['y'][i], 0, 1)
        if goodness > best:
            best = goodness
            bestI = i

    return {
        'value': region['value'],
        'x': region['x'][bestI],
        'y': region['y'][bestI]
    }

def getBelowValue(mat, region):
    x = region['x'][0]
    y = region['y'][0]
    while y < mat.shape[0] and mat[y, x] == region['value']:
        y += 1
    return mat[y][x] if y < mat.shape[0] else None

def removeRegion(mat, region, value):
    x_coords = region['x']
    y_coords = region['y']
    mat[y_coords, x_coords] = value

def getLabelLocs(mat):
    width = len(mat[0])
    height = len(mat)
    covered = [[False] * width] * height

    labelLocs = []
    for y in range(0, height):
        for x in range(0, width):
            if covered[y][x] == False:
                region = getRegion(mat, covered, x, y)
                coverRegion(covered, region)
            if len(region['x']) > 100:
                labelLocs.append(getLabelLoc(mat, region))
            else:
                removeRegion(mat, region)

    return labelLocs


def edge_mask(image, line_size=3, blur_value=9):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, line_size, blur_value)

    return edges


def merge_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def blur_image(image, blur_d=5):
    return cv2.bilateralFilter(image, d=blur_d, sigmaColor=200, sigmaSpace=200)
