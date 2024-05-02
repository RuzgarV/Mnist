from PIL import Image

image = Image.open('Adli4.jpg')

resized_image = image.resize((28, 28))

resized_image.save('mnist_formatinda_resim.jpg')
