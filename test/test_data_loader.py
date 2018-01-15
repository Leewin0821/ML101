from testtools import TestCase
from testtools.matchers import Equals

from data_loader import load_mnist


class TestDataLoader(TestCase):

    def test_data_load_correctly(self):
        self.images, self.labels = load_mnist('../mnist', kind='train')
        self.assertThat(self.images.shape, Equals((60000, 784)))
        self.assertThat(self.labels.shape, Equals((60000, 1)))
