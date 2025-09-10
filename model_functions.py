import numpy as np


# For scaling the features.
def scaled(matrix):
    """
    Args: 
    matrix    (np.ndarray)

    Returns:
    mat_scaled      (np.ndarray)
    mean            (np.ndarray)
    std             (np.ndarray)
    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std==0]=1e-8
    mat_scaled = (matrix - mean)/std
    return mat_scaled, mean, std



# For finding the cost at given values of w and b
def cost_logistic(x,y,w,b,lamb):
    """
    Args:
    x        (np.ndarray) 
    y        (np.ndarray)
    w        (np.ndarray)
    b        (float64)
    lamb     (float64)
    
    Returns:
    avg_cost     (float64)
    """
                   
    m = x.shape[0]
    z = np.dot(x,w)+b
    f = 1/(1+np.exp(-z))
    cost = np.sum(y * np.log(f) + (1-y) * np.log(1-f))
    regularization_term = (lamb/(2 * m)) * np.sum(np.square(w))

    avg_cost = (-1/m)*cost + regularization_term 

    return avg_cost



def gradient_descent(x_train,y_train,w_init,b_init,alpha,lamb,iterations,tolerance):
    """
    Args:
    x_train  (np.ndarray) 
    y_train  (np.ndarray)
    w_init   (np.ndarray)
    b_init   (float64)
    alpha    (float64)
    lamb     (float64)

    Returns:

    w             (np.ndarray)
    b             (float64)
    cost_history  (list)
    """

    m = x_train.shape[0]
    w = w_init
    b = b_init
    cost_history = []

    prev_cost = cost_logistic(x_train, y_train, w, b, lamb)
    cost_history.append(prev_cost)

    for i in range(iterations):

        z = np.dot(x_train,w)+b
        f = 1/(1+np.exp(-z))
        error = f - y_train

        dj_dw = (1/m) * np.dot(x_train.T , error) + (lamb/m) * w
        dj_db = (1/m) * np.sum(error)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        current_cost = cost_logistic(x_train,y_train,w,b,lamb)
        cost_history.append(current_cost)

        if abs(current_cost - prev_cost) <= tolerance:
            print(f'converged after {i + 1} iterations.')
            break
        prev_cost = current_cost

    return w, b, cost_history

