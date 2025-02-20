"""fedavg: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwr.common import parameters_to_ndarrays
from fedavg.task import load_data, load_model
from flwr.common import Parameters
import tensorflow as tf


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        parameters = tf.convert_to_tensor(parameters)
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print(f"Received parameters: {parameters}")  # Debugging
        print(f"Shapes of parameters:")
        for param in parameters:
            print(f"- {param.shape}")  # Check shape of each parameter
        parameters = tf.convert_to_tensor(parameters)  # Convert to tensor
        
        self.model.set_weights(parameters)

        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"mae": mae}


def client_fn(context: Context):
    # Load model and data
    net = load_model(input_size=1, hidden_size=64, output_size=1, dropout_rate=0.1)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
