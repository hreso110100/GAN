from src.gan.gan import GAN

if __name__ == '__main__':
    gan = GAN()

    gan.train(epochs=7000, batch_size=100, sample_interval=200)
