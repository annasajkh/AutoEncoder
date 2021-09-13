from libs.neural_network import *
from libs.activation_functions import *
import numpy as np
from PIL import Image
from tqdm import tqdm

nn = NeuralNetwork([
    LayerDense(784, 392, sigmoid),
    LayerDense(392, 196, leaky_relu),
    LayerDense(196, 98, sigmoid),
    LayerDense(98, 196, leaky_relu),
    LayerDense(196, 392, sigmoid),
    LayerDense(392, 784, sigmoid),
])

nn.set_learning_rate(0.001)

# 28x28 grayscale image of a cat doodle flatten to 1D
test_data = np.array(Image.open("example.png").resize((28, 28)).convert("L")).flatten() / 255

epoch = 20_000

for i in tqdm(range(epoch)):
    nn.train(test_data, test_data)


output = nn.forward(test_data)


out_img : Image = Image.new("L", (28,28))
pixels = out_img.load()

index = 0
for x in range(out_img.size[0]):
    for y in range(out_img.size[1]):
        pixels[y, x] = int(output[index] * 255)
        index+=1
out_img = out_img.resize((500,500), Image.ANTIALIAS)
out_img.show()


out_img : Image = Image.new("L", (28,28))
pixels = out_img.load()

index = 0
for x in range(out_img.size[0]):
    for y in range(out_img.size[1]):
        pixels[y, x] = int(test_data[index] * 255)
        index+=1
out_img = out_img.resize((500,500), Image.ANTIALIAS)
out_img.show()