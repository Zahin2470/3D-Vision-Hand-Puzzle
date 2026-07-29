import unittest
from unittest.mock import patch

from mediapipe.tasks.python import BaseOptions

from VisionPuzzle import tracker


class TrackerTests(unittest.TestCase):
    def test_mediapipe_delegate_prefers_cpu_on_macos(self) -> None:
        with patch("platform.system", return_value="Darwin"):
            delegate = tracker.get_mediapipe_delegate()

        self.assertIsNotNone(delegate)
        self.assertEqual(delegate, BaseOptions.Delegate.CPU)


if __name__ == "__main__":
    unittest.main()
