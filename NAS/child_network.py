import torch.nn as nn

class ChildNetwork(nn.Module):
    '''
        generic child network
    '''

    def __init__(self, actions):
        kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions

        self.features = nn.Sequential(
            nn.Conv2D(3, filters_1, (kernel_1, kernel_1), stride=1, padding=2),
            nn.BatchNorm2d(filters_1),
            nn.ReLU(),
            nn.Conv2D(filters_1, filters_2, (kernel_2, kernel_2), stride=1, padding=2),
            nn.BatchNorm2d(filters_2),
            nn.ReLU(),
            nn.Conv2D(filters_2, filters_3, (kernel_3, kernel_3), stride=1, padding=2),
            nn.BatchNorm2d(filters_3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (batch_size, channels)
        )

        self.classifier = nn.Sequential(
            nn.Linear(filters_3, 10)
        )

    def forward(self, input):
        bsize = input.size(0)
        output = self.features(input)

        output = output.view(bsize, -1)  # flatten 2D feature maps into a 1D vector for each input
        output = self.classifier(output)
        return output

