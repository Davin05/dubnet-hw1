#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// tensor x: input to layer
// returns: the result of running the layer y = f(x)
tensor forward_activation_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    ACTIVATION a = l->activation;
    tensor y = tensor_copy(x);

    // TODO: 2.0
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 

    assert(x.n >= 2);
    float* data = y.data;
    if(a == LOGISTIC){
        size_t sizeY = tensor_len(y);
        for(int i = 0; i < sizeY; i++){
            data[i] = (float) 1.0/ (1.0 + (float) exp(-1.0* data[i]));
        }
    } else if(a == RELU){
        size_t sizeY = tensor_len(y);

        for(int i = 0; i < sizeY; i++){
            data[i] = (data[i]< 0? 0: data[i]);
        }
    } else if(a == LRELU){
        size_t sizeY = tensor_len(y);
        for(int i = 0; i < sizeY; i++){
            data[i] = (data[i]< 0? 0.01 * data[i]: data[i]);
        }
    } else { // It's soft max
        /*
        for each row (num rows is everything in dim *= eachother)
            sum the row
            do the soft max?
        */
       size_t numRow = y.size[0];
       size_t numCol = y.size[1];
       for(int i = 2; i < y.n; i++){
        numRow*= y.size[i];
       }
       for(int i = 0; i < numRow; i++){
        size_t rowStart = i*numCol;
        float sum = 0;
        for(int j = rowStart; j < rowStart+numCol; j++){
            sum += (float) exp(data[j]);
        }
        for(int j = rowStart; j < rowStart+numCol; j++){
            data[j] = (float) exp(data[j])/sum;
        }
       }

    }
    /* You might want this
    size_t i, j;
    for(i = 0; i < x.size[0]; ++i){
        tensor x_i = tensor_get_(x, i);
        tensor y_i = tensor_get_(y, i);
        size_t len = tensor_len(x_i);
        // Do stuff in here
    }
    */

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
tensor backward_activation_layer(layer *l, tensor dy)
{
    tensor x = l->x;
    tensor dx = tensor_copy(dy);
    ACTIVATION a = l->activation;

    // TODO: 2.1
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1

    /* Might want this too
    size_t i, j;
    for(i = 0; i < dx.size[0]; ++i){
        tensor x_i = tensor_get_(x, i);
        tensor dx_i = tensor_get_(dx, i);
        size_t len = tensor_len(dx_i);
        // Do stuff in here
    }
    */
    float* dataX= x.data;
    float* dataDy = dy.data;
    float* dataDx = dx.data;
    if(a == LOGISTIC){
        size_t sizeX = tensor_len(x);
        for(int i = 0; i < sizeX; i++){
            float logx = (float) 1.0/ (1 + (float) exp(-1.0* dataX[i]));
            dataDx[i] = dataDy[i] * logx * (1.0-logx);
        }
    } else if(a == RELU){
        size_t sizeX = tensor_len(x);
        for(int i = 0; i < sizeX; i++){
            dataDx[i] = dataDy[i] * (dataX[i] > 0 ? 1 : 0);
        }
    } else if(a == LRELU){
        size_t sizeX = tensor_len(x);
        for(int i = 0; i < sizeX; i++){
            dataDx[i] = dataDy[i] * (dataX[i] > 0 ? 1 : 0.01);
        }
    } else { // It's soft max
        // do nothing
    }
    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer *l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
