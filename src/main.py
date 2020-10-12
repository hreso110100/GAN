from src.gan.gan import GAN

if __name__ == '__main__':
    gan = GAN()

    gan.train(epochs=5000, batch_size=256, sample_interval=200)
