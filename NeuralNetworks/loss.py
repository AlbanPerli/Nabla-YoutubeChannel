def MeanSquareError(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    sum = 0
    for y, y_pred in zip(y_true, y_pred):
         sum = (y - y_pred) ** 2
    return sum / len(y_true)

def d_MeanSquareError(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    grads = []
    n = len(y_true)
    for y, y_pred in zip(y_true, y_pred):
        grads.append( (-2 * (y - y_pred)) / n )
    return grads

