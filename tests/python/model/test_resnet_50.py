import mnm
from mnm.model import Conv2d, Linear, BatchNorm, Sequential


class BottleNeck(mnm.Model):
    expansion = 4

    # pylint: disable=attribute-defined-outside-init
    def build(self, in_planes, planes, stride=1):
        self.bn1 = BatchNorm(in_planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.conv3 = Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Conv2d(in_planes, self.expansion * planes,
                                   kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None
    # pylint: enable=attribute-defined-outside-init

    @mnm.model.script
    def forward(self, x):
        out = mnm.relu(self.bn1(x))

        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(mnm.relu(self.bn2(out)))
        out = self.conv3(mnm.relu(self.bn3(out)))
        out = mnm.add(out, shortcut)

        return out


class ResNet50(mnm.Model):

    # pylint: disable=attribute-defined-outside-init
    def build(self, num_blocks, num_classes=10):
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = Linear(512 * BottleNeck.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for one_stride in strides:
            layers.append(BottleNeck(self.in_planes, planes, one_stride))
            self.in_planes = planes * BottleNeck.expansion

        return Sequential(*layers)
    # pylint: enable=attribute-defined-outside-init

    @mnm.model.script
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = mnm.avg_pool2d(out, 4)
        out = mnm.batch_flatten(out)
        out = self.linear(out)

        return out


def test_build():
    x = mnm.array([1, 2, 3], dtype="float32", ctx="cpu")
    model = ResNet50([3, 4, 6, 3])
    print("### Switch to training mode")
    model.train_mode()
    model(x)
    model(x)
    model(x)
    model(x)
    print("### Switch to infer mode")
    model.infer_mode()
    model(x)


if __name__ == "__main__":
    test_build()
