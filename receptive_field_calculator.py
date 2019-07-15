import math
import numpy as np

# [ksize, stride, padding]
convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']

# image input size
imsize = 227


def outFromIn(conv, layerIn):

    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]

    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = np.floor((n_in - k + 2*p)/s) + 1
    actualP = (n_out - 1)*s - n_in + k
    pR = np.ceil(actualP)
    pL = np.floor(actualP/2)

    j_out = j_in * s
    r_out = r_in * (k - 1) * j_in
    start_out = start_in + ((k - 1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
    print(layer_name + ':')
    print(print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3])))

layerInfos = []
if __name__ == '__main__':

    print('-----Net Summary-----')
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, 'input_image')
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])

    print('----------------------')
    layer_name = "Layer name where the feature in:"
    layer_idx = layer_names.index(layer_name)
    idx_x = int("index of the feature in x dimension (from 0)")
    idx_y = int("index of the feature in y dimension (from 0)")

    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]

    assert(idx_x < n)
    assert(idx_y < n)

    print('receptive field: ({}, {})'.format(r, r))
    print('center: ({}, {})'.format(start+idx_x*j, start+idx_y*j))