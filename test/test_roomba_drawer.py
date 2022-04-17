from roomba_drawer import (
    RobotMoveGenerator,
    COMMAND_ROTATE,
    COMMAND_MOVE,
    COMMAND_MARKER_DOWN,
    COMMAND_MARKER_UP,
)
import math
import unittest


class RobotDrawerTest(unittest.TestCase):
    def test_no_rot_stroke(self):
        move_generator = RobotMoveGenerator()
        stroke = [[0, 0], [1, 0]]

        moves = move_generator.draw_stroke(stroke)
        self.assertEqual(moves[0][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[0][1], 0.0)

        self.assertEqual(moves[1][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[1][1], 0.0)

        self.assertEqual(moves[2][0], COMMAND_MARKER_DOWN)

        self.assertEqual(moves[3][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[3][1], 0.0)

        self.assertEqual(moves[4][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[4][1], 1.0)

        self.assertEqual(moves[5][0], COMMAND_MARKER_UP)

    def test_45_deg_up(self):
        move_generator = RobotMoveGenerator()
        stroke = [[0, 0], [1, 1]]

        moves = move_generator.draw_stroke(stroke)
        self.assertEqual(moves[0][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[0][1], 0.0)

        self.assertEqual(moves[1][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[1][1], 0.0)

        self.assertEqual(moves[2][0], COMMAND_MARKER_DOWN)

        self.assertEqual(moves[3][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[3][1], math.pi / 4)

        self.assertEqual(moves[4][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[4][1], math.sqrt(2))

        self.assertEqual(moves[5][0], COMMAND_MARKER_UP)

    def test_45_deg_down_and_back(self):
        move_generator = RobotMoveGenerator()
        stroke = [[0, 0], [1, 1], [0, 0]]

        moves = move_generator.draw_stroke(stroke)
        self.assertEqual(moves[0][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[0][1], 0.0)

        self.assertEqual(moves[1][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[1][1], 0.0)

        self.assertEqual(moves[2][0], COMMAND_MARKER_DOWN)

        self.assertEqual(moves[3][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[3][1], math.pi / 4)

        self.assertEqual(moves[4][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[4][1], math.sqrt(2))

        self.assertEqual(moves[5][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[5][1], math.pi)

        self.assertEqual(moves[6][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[6][1], math.sqrt(2))

        self.assertEqual(moves[7][0], COMMAND_MARKER_UP)

    def test_square(self):
        move_generator = RobotMoveGenerator()
        stroke = [[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]
        moves = move_generator.draw_stroke(stroke)

        self.assertEqual(moves[0][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[0][1], math.pi / 4)

        self.assertEqual(moves[1][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[1][1], math.sqrt(2))

        self.assertEqual(moves[2][0], COMMAND_MARKER_DOWN)

        self.assertEqual(moves[3][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[3][1], -3 * math.pi / 4)

        self.assertEqual(moves[4][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[4][1], 2)

        self.assertEqual(moves[5][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[5][1], -math.pi / 2)

        self.assertEqual(moves[6][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[6][1], 2)

        self.assertEqual(moves[7][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[7][1], -math.pi / 2)

        self.assertEqual(moves[8][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[8][1], 2)

        self.assertEqual(moves[9][0], COMMAND_ROTATE)
        self.assertAlmostEqual(moves[9][1], -math.pi / 2)

        self.assertEqual(moves[10][0], COMMAND_MOVE)
        self.assertAlmostEqual(moves[10][1], 2)

        self.assertEqual(moves[11][0], COMMAND_MARKER_UP)
