import unittest
from app.main import Main


class WeChatClientTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_main(self):
        main = Main()
        self.assertIsNotNone(main.get_name())