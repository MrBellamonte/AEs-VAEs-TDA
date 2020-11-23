from scripts.ssc.models.Competitors import (
    mnist_test)
from src.competitors.train_engine import simulator_competitor

if __name__ == "__main__":
    simulator_competitor(mnist_test)


