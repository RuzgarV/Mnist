from PIL import Image

# Resmi yükle
image = Image.open('Adli4.jpg')

# Resmi 28x28 boyutuna yeniden boyutlandır
resized_image = image.resize((28, 28))

# Sonucu kaydet
resized_image.save('mnist_formatinda_resim.jpg')
