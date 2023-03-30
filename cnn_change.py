import os, cv2, mnist, numpy as np
import cloudpickle as pickle
class Softmax:
    def __init__(self, input_len=26 * 26 * 10, nodes=10):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)
    def forward_softmax(self, input):
        self.last_input_shape = input.shape
        self.last_input = input.flatten()
        input_len, nodes = self.weights.shape
        totals = np.dot(input.flatten(), self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
    def backprop_softmax(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)
class Conv3x3:
    def __init__(self, num_filters=10):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
    def forward_conv(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output
    def backprop_conv(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_L_d_filters
        return None
train_images, train_labels = mnist.train_images()[:3_000], mnist.train_labels()[:3_000]
test_images, test_labels = mnist.test_images()[:3_000], mnist.test_labels()[:3_000]
conv = Conv3x3()  # 28x28x1 -> 26x26x8
softmax = Softmax()  # 26x26x8 -> 10
def forward(image, label):
    out = conv.forward_conv((image / 255))
    out = softmax.forward_softmax(out)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc
def predict(image,conv,softmax):
#    cv2.imshow("",(image / 255) - 0.5)
#    cv2.waitKey(0)
    out = conv.forward_conv(image/255)
    out = softmax.forward_softmax(out)
    print("Guess is:", np.argmax(out))
    return np.argmax(out)
def train(im, label, lr=0.001):
    out, loss, acc = forward(im, label)
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    gradient = softmax.backprop_softmax(gradient, lr)
    gradient = conv.backprop_conv(gradient, lr)
    return loss, acc
def train_model():
    for epoch in range(1):
        print('--- Epoch %d ---' % (epoch + 1))
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 99:
                print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, num_correct))
                loss = 0
                num_correct = 0
            l, acc = train(im, label)
            loss += l
            num_correct += acc
def check_model_files():
    return False if not os.path.exists('conv') or not os.path.exists('softmax') else True
def save_model():
    with open('conv', 'wb') as f_conv, open('softmax', 'wb') as f_soft:
        pickle.dump(conv, f_conv), pickle.dump(softmax, f_soft)
def load_model():
    global conv, softmax
    with open(os.getcwd()+'\conv', 'rb') as f_conv, open(os.getcwd()+'\softmax', 'rb') as f_soft:
        conv, softmax = pickle.load(f_conv), pickle.load(f_soft)
if __name__ == '__main__':
    if check_model_files():
        print("loading the model...")
        load_model()
    else:
        print("training the model from scratch...")
        train_model()
        save_model()
    for i in range(10):
        predict(cv2.imread(os.getcwd()+"\\"+str(i)+".png", cv2.IMREAD_GRAYSCALE),conv,softmax)