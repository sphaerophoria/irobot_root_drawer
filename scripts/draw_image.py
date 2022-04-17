import asyncio
import roomba_drawer
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import math
import cv2
import aiorobot


class OpenCVCommandRunner:
    """
    Pretends to be a robot but just draws lines on a canvas with opencv instead
    of issuing commands to the robot.
    """

    def __init__(self, canvas_size):
        self.canvas = np.zeros((canvas_size[1], canvas_size[0], 3))
        self.position = np.array([canvas_size[0] / 2, canvas_size[1] / 2])
        self.rotation_rad = 0
        self.marker_is_down = False

    def issue_command(self, command):
        if command[0] == roomba_drawer.COMMAND_MARKER_UP:
            self.marker_up()
        elif command[0] == roomba_drawer.COMMAND_MARKER_DOWN:
            self.marker_down()
        elif command[0] == roomba_drawer.COMMAND_MOVE:
            self.move(command[1])
        elif command[0] == roomba_drawer.COMMAND_ROTATE:
            self.rotate(command[1])
        else:
            raise RuntimeError(f"Unknown command: {command[0]}")

    def marker_down(self):
        self.marker_is_down = True

    def marker_up(self):
        self.marker_is_down = False

    def move(self, distance):
        xoffset = math.cos(self.rotation_rad) * distance
        yoffset = math.sin(self.rotation_rad) * distance
        end_pos = self.position + [xoffset, yoffset]

        if self.marker_is_down:
            cv2.line(
                self.canvas,
                self.position.astype(int),
                end_pos.astype(int),
                color=(1, 1, 1, 0),
            )
        self.position = end_pos

    def rotate(self, amount):
        self.rotation_rad += amount


async def run_commands(moves):
    async with aiorobot.get_robot() as robot:
        robot_generator = roomba_drawer.RobotCommandRunner(robot)
        for move in moves:
            await robot_generator.issue_command(move)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--intermediate-output-dir", type=Path, default=None)
    parser.add_argument("--simulate", action="store_true", default=None)

    return parser.parse_args(args)


def main(image, simulate, intermediate_output_dir):
    strokes = roomba_drawer.generate_strokes(image, intermediate_output_dir)

    if intermediate_output_dir is not None:
        roomba_drawer.save_strokes(strokes, intermediate_output_dir / "strokes.png")

    # Generate the moves before we actually run them. This decouples implementation of
    # the robot API from the math to generate the moves, which also makes it easier
    # to unit test move generation without any mocks
    move_generator = roomba_drawer.RobotMoveGenerator()

    moves = []
    for stroke in strokes:
        moves.extend(move_generator.draw_stroke(stroke))

    if simulate:
        opencv_generator = OpenCVCommandRunner((100, 100))
        for move in moves:
            opencv_generator.issue_command(move)

        cv2.imshow("Simulated result", opencv_generator.canvas)
        cv2.waitKey()
    else:
        asyncio.run(run_commands(moves))


if __name__ == "__main__":
    main(**vars(parse_args()))
