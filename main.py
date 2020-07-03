from data_loading import load_train_test
from custom_model import CustomCNN
from transfer_learning import VGGBasedModel


train_x, train_y, test_x, test_y = load_train_test()
cnn_model = CustomCNN(img_size=512)
cnn_model.train(train_x, train_y)
print("Custom CNN model accuracy: {}".format(cnn_model.evaluate(test_x, test_y)))

vgg_model = VGGBasedModel(img_size=512)
vgg_model.train(train_x, train_y)
print("VGG based model accuracy: {}".format(vgg_model.evaluate(test_x, test_y)))
