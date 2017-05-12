#Source: https://github.com/mfaruqui/dwl/blob/master/src/rprop.py
import numpy
import math
import theano
import theano.tensor as T
theano.config.floatX='float32'
def rprop_plus_updates(params, grads):

    # RPROP+ parameters
    updates = []
    deltas = 0.1*numpy.ones(len(params))
    last_weight_changes = numpy.zeros(len(params))
    last_params = params
    
    positiveStep = 1.2
    negativeStep = 0.5
    maxStep = 50.
    minStep = math.exp(-6)

    # RPROP+ parameter update (original Reidmiller implementation)
    for param, gparam, last_gparam, delta, last_weight_change in \
            zip(params, grads, last_params, deltas, last_weight_changes):
        # calculate change
        change = T.sgn(gparam * last_gparam)
        if T.gt(change, 0) :
            delta = T.minimum(delta * positiveStep, maxStep)
            weight_change = T.sgn(gparam) * delta
            last_gparam = gparam
            
        elif T.lt(change, 0):
            delta = T.maximum(delta * negativeStep, minStep)
            weight_change = -last_weight_change
            last_gparam = 0
            
        else:
            weight_change = T.sgn(gparam) * delta
            last_gparam = param

        # update the weights
        updates.append((param, param - weight_change))
        # store old change
        last_weight_change = weight_change

    return updates

def irprop_minus_updates(params, grads):

    # IRPROP- parameters
    updates = []
    deltas = 0.1*numpy.ones(len(params),theano.config.floatX)
    last_params = params
    
    positiveStep = 1.2
    negativeStep = 0.5
    maxStep = 50#1.
    minStep = math.exp(-6)

    for param, gparam, delta, last_gparam in zip(params, grads, deltas, last_params):
        # calculate change
        change = T.sgn(gparam * last_gparam)
        if T.gt(change, 0) :
            delta = T.minimum(delta * positiveStep, maxStep)
                           
        elif T.lt(change, 0):
            delta = T.maximum(delta * negativeStep, minStep)
            
            last_gparam = 0
        delta = delta.astype('float32')
            
        # update the weights
        updates.append((param, param - T.sgn(gparam) * delta))
        # store old change
        last_gparam = gparam

    return updates
