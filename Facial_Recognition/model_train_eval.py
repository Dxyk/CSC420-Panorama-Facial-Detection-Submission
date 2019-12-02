import torch
import torch.nn as nn
import torchvision
import cv2
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt

# ========== CONSTANTS ==========
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
actor_names = ["Aaron Paul", "Amaury Nolasco", "Anna Gunn", "bill gates",
               "Brit Marling", "Bryan Cranston", "Caity Lotz", "rihanna",
               "tati gabrielle", "Wentworth Miller"]
# actor genders: 0: female, 1: male
actor_genders = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]  # correspond to names


def generate_sets(actor):
    image_list = [f_name for f_name in os.listdir("./Resource/cropped_227/")
                  if actor in f_name]

    np.random.seed(0)
    np.random.shuffle(image_list)

    x_train = np.zeros((0, 3, 227, 227))
    x_val = np.zeros((0, 3, 227, 227))
    x_test = np.zeros((0, 3, 227, 227))

    for i in range(len(image_list)):
        img = cv2.imread("./Resource/cropped_227/" + image_list[i])
        img = img[:, :, :3]
        img = img / 128. - 1.
        img = np.rollaxis(img, -1).astype(np.float32)
        try:
            img = np.reshape(img, [1, 3, 227, 227])
        except Exception:
            print(image_list[i])
        if i in range(10):
            x_test = np.vstack((x_test, img))
        elif i in range(10, 20):
            x_val = np.vstack((x_val, img))
        else:
            x_train = np.vstack((x_train, img))

    return x_train, x_val, x_test


class MyAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        built_in_alex_net = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = built_in_alex_net.features[i].weight
            self.features[i].bias = built_in_alex_net.features[i].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = built_in_alex_net.classifier[i].weight
            self.classifier[i].bias = built_in_alex_net.classifier[i].bias

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 13 * 13)
        return x


class IdentityRecognitionNet(nn.Module):
    def __init__(self, dim_x, dim_h, dim_out):
        super(IdentityRecognitionNet, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.linear(x)


class GenderClassificationNet(nn.Module):
    def __init__(self, dim_x, dim_h):
        super(GenderClassificationNet, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, 2),
        )

    def forward(self, x):
        return self.linear(x)


def train_identity_nn(dim_h, alpha, epoch):
    alex_model = MyAlexNet()
    print(alex_model)
    alex_model.eval()
    # hyper-param
    torch.manual_seed(5)
    dim_x = 256 * 13 * 13
    dim_out = len(actor_names)

    identityNet = IdentityRecognitionNet(dim_x, dim_h, dim_out)
    print(identityNet)

    # get data
    print("========== Fetching Data ==========")
    x_train, y_train = np.zeros((0, 3, 227, 227)), np.zeros(
        (0, len(actor_names)))
    x_val, y_val = np.zeros((0, 3, 227, 227)), np.zeros((0, len(actor_names)))
    x_test, y_test = np.zeros((0, 3, 227, 227)), np.zeros((0, len(actor_names)))

    for i in range(len(actor_names)):
        a_name = actor_names[i]
        print("===")
        print(a_name)

        x_train_i, x_val_i, x_test_i = generate_sets(a_name)
        print(x_train_i.shape)
        print(x_val_i.shape)
        print(x_test_i.shape)

        one_hot = np.zeros(len(actor_names))
        one_hot[i] = 1

        x_train = np.vstack((x_train, x_train_i))
        x_val = np.vstack((x_val, x_val_i))
        x_test = np.vstack((x_test, x_test_i))

        y_train = np.vstack(
            (y_train, np.tile(one_hot, (x_train_i.shape[0], 1))))
        y_val = np.vstack((y_val, np.tile(one_hot, (x_val_i.shape[0], 1))))
        y_test = np.vstack((y_test, np.tile(one_hot, (x_test_i.shape[0], 1))))

    train_activation = np.zeros((0, 256 * 13 * 13))
    val_activation = np.zeros((0, 256 * 13 * 13))
    test_activation = np.zeros((0, 256 * 13 * 13))

    print(x_train.shape)
    for i in range(x_train.shape[0] // 100 + 1):
        x = Variable(torch.from_numpy(x_train[100 * i: 100 * (i + 1)]),
                     requires_grad=False).type(dtype_float)
        train_activation = np.vstack(
            (train_activation, alex_model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_val), requires_grad=False).type(dtype_float)
    val_activation = np.vstack(
        (val_activation, alex_model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_test), requires_grad=False).type(
        dtype_float)
    print(x.shape)
    test_activation = np.vstack(
        (test_activation, alex_model.forward(x).data.numpy()))

    # train_identity
    print("========== Start Training ==========")
    x = Variable(torch.from_numpy(train_activation), requires_grad=False).type(
        dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(y_train, 1)),
                         requires_grad=False).type(dtype_long)

    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = []
    train_perf = []
    val_perf = []
    test_perf = []

    optimizer = torch.optim.Adam(identityNet.parameters(), lr=alpha)
    for i in range(epoch):
        y_pred = identityNet(x)
        loss = loss_fn(y_pred, y_classes)

        identityNet.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == epoch - 1:
            print("\nIteration {}".format(i))
            epochs.append(i)

            x_train = Variable(torch.from_numpy(train_activation),
                               requires_grad=False).type(dtype_float)
            y_pred = identityNet(x_train).data.numpy()
            train_perf_i = (np.mean(
                np.argmax(y_pred, 1) == np.argmax(y_train, 1))) * 100
            print("Training: {}%".format(train_perf_i))
            train_perf.append(train_perf_i)

            x_val = Variable(torch.from_numpy(val_activation),
                             requires_grad=False).type(dtype_float)
            y_pred = identityNet(x_val).data.numpy()
            val_perf_i = (np.mean(
                np.argmax(y_pred, 1) == np.argmax(y_val, 1))) * 100
            print("Validation: {}%".format(val_perf_i))
            val_perf.append(val_perf_i)

            x_test = Variable(torch.from_numpy(test_activation),
                              requires_grad=False).type(dtype_float)
            y_pred = identityNet(x_test).data.numpy()
            print(y_pred)
            test_perf_i = (np.mean(
                np.argmax(y_pred, 1) == np.argmax(y_test, 1))) * 100
            print("Testing: {}%".format(test_perf_i))
            test_perf.append(test_perf_i)

    torch.save(identityNet.state_dict(),
               "model/identityNet.pth")
    torch.save(alex_model.state_dict(), "model/alex_model.pth")
    return epochs, train_perf, val_perf, test_perf


def train_gender_nn(dim_h, alpha, epoch):
    alex_model = MyAlexNet()
    print(alex_model)
    alex_model.eval()
    # hyper-param
    torch.manual_seed(5)
    dim_x = 256 * 13 * 13

    genderNet = GenderClassificationNet(dim_x, dim_h)
    print(genderNet)

    # get data
    print("========== Fetching Data ==========")
    x_train, y_train = np.zeros((0, 3, 227, 227)), np.zeros(
        (0, 2))
    x_val, y_val = np.zeros((0, 3, 227, 227)), np.zeros((0, 2))
    x_test, y_test = np.zeros((0, 3, 227, 227)), np.zeros((0, 2))

    for i in range(len(actor_names)):
        a_name = actor_names[i]
        print("===")
        print(a_name)

        x_train_i, x_val_i, x_test_i = generate_sets(a_name)
        print(x_train_i.shape)
        print(x_val_i.shape)
        print(x_test_i.shape)
        #
        # x_train = np.vstack((x_train, x_train_i))
        # x_val = np.vstack((x_val, x_val_i))
        # x_test = np.vstack((x_test, x_test_i))

        one_hot = np.zeros(2)
        one_hot[actor_genders[i]] = 1

        x_train = np.vstack((x_train, x_train_i))
        x_val = np.vstack((x_val, x_val_i))
        x_test = np.vstack((x_test, x_test_i))

        y_train = np.vstack(
            (y_train, np.tile(one_hot, (x_train_i.shape[0], 1))))
        y_val = np.vstack((y_val, np.tile(one_hot, (x_val_i.shape[0], 1))))
        y_test = np.vstack((y_test, np.tile(one_hot, (x_test_i.shape[0], 1))))
        #
        # y_train = np.vstack(
        #     (y_train, np.tile(actor_genders[i], (x_train_i.shape[0], 1))))
        # y_val = np.vstack(
        #     (y_val, np.tile(actor_genders[i], (x_val_i.shape[0], 1))))
        # y_test = np.vstack(
        #     (y_test, np.tile(actor_genders[i], (x_test_i.shape[0], 1))))

    train_activation = np.zeros((0, 256 * 13 * 13))
    val_activation = np.zeros((0, 256 * 13 * 13))
    test_activation = np.zeros((0, 256 * 13 * 13))

    print(x_train.shape)
    for i in range(x_train.shape[0] // 100 + 1):
        x = Variable(torch.from_numpy(x_train[100 * i: 100 * (i + 1)]),
                     requires_grad=False).type(dtype_float)
        train_activation = np.vstack(
            (train_activation, alex_model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_val), requires_grad=False).type(dtype_float)
    val_activation = np.vstack(
        (val_activation, alex_model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_test), requires_grad=False).type(
        dtype_float)
    print(x.shape)
    test_activation = np.vstack(
        (test_activation, alex_model.forward(x).data.numpy()))

    # train_identity
    print("========== Start Training ==========")
    x = Variable(torch.from_numpy(train_activation), requires_grad=False).type(
        dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(y_train, 1)),
                         requires_grad=False).type(dtype_long)

    loss_fn = torch.nn.BCELoss()

    epochs = []
    train_perf = []
    val_perf = []
    test_perf = []

    optimizer = torch.optim.Adam(genderNet.parameters(), lr=alpha)
    for i in range(epoch):
        y_pred = genderNet(x)
        loss = loss_fn(y_pred, y_classes)

        genderNet.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == epoch - 1:
            print("\nIteration {}".format(i))
            epochs.append(i)

            x_train = Variable(torch.from_numpy(train_activation),
                               requires_grad=False).type(dtype_float)
            y_pred = genderNet(x_train).data.numpy()
            train_perf_i = (np.mean(
                np.argmax(y_pred, 1) == np.argmax(y_train, 1))) * 100
            print("Training: {}%".format(train_perf_i))
            train_perf.append(train_perf_i)

            x_val = Variable(torch.from_numpy(val_activation),
                             requires_grad=False).type(dtype_float)
            y_pred = genderNet(x_val).data.numpy()
            val_perf_i = (np.mean(
                np.argmax(y_pred, 1) == np.argmax(y_val, 1))) * 100
            print("Validation: {}%".format(val_perf_i))
            val_perf.append(val_perf_i)

            x_test = Variable(torch.from_numpy(test_activation),
                              requires_grad=False).type(dtype_float)
            y_pred = genderNet(x_test).data.numpy()
            print(y_pred)
            test_perf_i = (np.mean(
                np.argmax(y_pred, 1) == np.argmax(y_test, 1))) * 100
            print("Testing: {}%".format(test_perf_i))
            test_perf.append(test_perf_i)

    torch.save(genderNet.state_dict(), "model/genderNet.pth")
    torch.save(alex_model.state_dict(), "model/alex_model.pth")
    return epochs, train_perf, val_perf, test_perf


def train_identity():
    alpha = 1e-4
    epoch = 180
    dim_h = 600

    epochs, train_results, val_results, test_results = train_identity_nn(dim_h,
                                                                         alpha,
                                                                         epoch)

    plt.plot(epochs, train_results, 'r-', label='Training Result')
    plt.plot(epochs, val_results, 'g-', label='Validation Result')
    plt.plot(epochs, test_results, 'b-', label='Test Result')
    plt.title('Identity Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.savefig("out/identity_training_curve.png")
    plt.close()
    return


def train_gender():
    alpha = 1e-4
    epoch = 80
    dim_h = 500

    epochs, train_results, val_results, test_results = train_gender_nn(dim_h,
                                                                       alpha,
                                                                       epoch)

    plt.plot(epochs, train_results, 'r-', label='Training Result')
    plt.plot(epochs, val_results, 'g-', label='Validation Result')
    plt.plot(epochs, test_results, 'b-', label='Test Result')
    plt.title('Gender Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.savefig("out/gender_training_curve.png")
    plt.close()
    return


def evaluate(img):
    dim_x = 256 * 13 * 13
    dim_h_id = 600
    dim_h_gender = 500
    dim_out = len(actor_names)
    alex_model = MyAlexNet()
    identity_model = IdentityRecognitionNet(dim_x, dim_h_id, dim_out)
    gender_model = GenderClassificationNet(dim_x, dim_h_gender)
    alex_model.load_state_dict(torch.load("model/alex_model.pth"))
    identity_model.load_state_dict(torch.load("model/identityNet.pth"))
    gender_model.load_state_dict(torch.load("model/genderNet.pth"))
    alex_model.eval()
    identity_model.eval()
    gender_model.eval()

    # Sample test
    img = img[:, :, :3]
    img = img / 128. - 1.
    img = np.rollaxis(img, -1).astype(np.float32)
    x_test = np.reshape(img, [1, 3, 227, 227])

    x = Variable(torch.from_numpy(x_test), requires_grad=False).type(
        dtype_float)
    print(x.shape)
    test_activation = np.zeros((0, 256 * 13 * 13))
    test_activation = np.vstack(
        (test_activation, alex_model.forward(x).data.numpy()))

    x_test = Variable(torch.from_numpy(test_activation),
                      requires_grad=False).type(dtype_float)
    y_identity_pred = identity_model(x_test).data.numpy()
    print(y_identity_pred)
    print("y_identity_pred: {}".format(
        actor_names[np.argmax(y_identity_pred, 1)[0]]))
    y_gender_pred = gender_model(x_test).data.numpy()
    print(y_gender_pred)
    gender = "male" if np.argmax(y_gender_pred, 1)[0] == 1 else "female"
    print("y_gender_pred: {}".format(gender))
    return actor_names[np.argmax(y_identity_pred, 1)[0]], gender


if __name__ == "__main__":
    # https://www.popsugar.com/celebrity/photo-gallery/46730684/image/46730686/el-camino-breaking-bad-movie-premiere-pictures

    # train_identity()
    train_gender()

    for img_name in ["failure_face1.jpg", "failure_face2.jpg",
        "failure_face3.jpg", "failure_face4.jpg"]:
        print(img_name)
        img = cv2.imread("./Resource/test_images/" + img_name)
        evaluate(img)
