import math

# parameters
w1, b1 = 0.5, 0.1
w2, b2 = -0.4, 0.1

lr = 0.01

def softmax(z1, z2):
    m = max(z1, z2)
    e1 = math.exp(z1 - m)
    e2 = math.exp(z2 - m)
    s = e1 + e2
    return e1/s, e2/s


# ---- forward ----
def feed_forward(x_input):
    z1 = w1*x_input + b1
    z2 = w2*x_input + b2

    p1, p2 = softmax(z1, z2)
    return p1, p2


def get_loss(model_output, y_true):
    if y_true[0] == 1:
        return -math.log(model_output[0])
    return -math.log(model_output[1])


# ---- backward ----
def backward(model_output, x, y_true):
    dz1 = model_output[0] - y_true[0]
    dz2 = model_output[1] - y_true[1]

    dw1 = dz1 * x
    db1 = dz1

    dw2 = dz2 * x
    db2 = dz2

    return dw1, db1, dw2, db2


def update_params(dw1, db1, dw2, db2):
    global w1, b1, w2, b2

    w1 -= lr * dw1
    b1 -= lr * db1

    w2 -= lr * dw2
    b2 -= lr * db2


# data
x = [10.0, 110.0, 120.0, 30.0, 40.0, 200.0]
y = [[1,0], [0,1], [0,1], [1,0], [1,0], [0,1]]


def main():
    for epoch in range(500):
        for x_input, y_true in zip(x, y):

            x_scaled = x_input

            p1, p2 = feed_forward(x_scaled)

            loss = get_loss((p1,p2), y_true)

            dw1, db1, dw2, db2 = backward((p1,p2), x_scaled, y_true)

            update_params(dw1, db1, dw2, db2)

        if epoch % 100 == 0:
            print("epoch:", epoch)
            print("loss:", loss)
            print("params:", w1, b1, w2, b2)
            print("prediction:", p1, p2)
            print()


main()
