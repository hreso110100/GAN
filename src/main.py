from src.gan.gan import GAN

if __name__ == '__main__':
    gan = GAN()

    gan.train(epochs=1000, batch_size=1, sample_interval=200)
    gan.plot_loss()
    gan.save_models()
