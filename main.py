import numpy as np
import matplotlib.pyplot as plt



class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights =  np.array([np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_price):
        # fais la difference entre l'input et les poids pour les comparer + un biais pour pouvoir influencer l'apprentissage
        print("bias:", self.bias)
        print("weights:", self.weights)
        print("input:", input_price)

        layer_1 = (self.weights / input_price) + self.bias
        # utilise la sigmoide pour reduire les images possibles dans le domaine [0;1]
        # car on a besoin de valeur qui soit 1 ou 0 pour l'apprentissage
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2  # juste histoire de bien nommer

        print("layer1:", layer_1)
        print("predict:", prediction)

        return prediction

    def _compute_gradients(self, input_price, target):
        # meme réseau neuronal que pour la prediction
        layer_1 = (input_price - self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        # on as besoin du calcul de chaque layer pour en faire leur dérivée et avoir mdes dérivés partielles du réseau neuronal
        derror_dprediction = 2 * (prediction - target)  # dérivée de l'erreur quadratique (x^2)'= 2*x
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)  # dérivée de la sigmoide (relation sur internet)
        dlayer1_dbias = 1  # regle des puissance (j'ai pas tout compris)
        dlayer1_dweights = (0 * self.weights) + (1 * input_price)  # on dérive aussi un entier et un facteur de degrés 1
        # on somme les dérivées partielles de chaque couche de calcul
        derror_dbias = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        # parreil
        derror_dweights = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )
        return derror_dbias, derror_dweights


    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
                derror_dweights * self.learning_rate
        )

    def train(self, input_price, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_price))  # crée un tableau rand avec autant de valeurs que le tableau en parametres

            input_vector = input_price[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_price)):
                    data_point = input_price[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    print(f"error:",error)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
input_vectors = np.array(
    [
        [33],
        [212],
        [43],
        [3456],
        [3.5],
        [232],
        [-53],
        [134],
        [112],
        [32345678],
        [-100],
        [20],
        [30000],
        [0],
        [-1234567],
        [-567],
        [12],
        [65],
        [150],
        [110],
        [11]
    ]
)
tar= np.array(len(input_vectors))

targets = np.array([1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.74])

learning_rate = 1

neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
#predict = neural_network.predict(1)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
plt.show()