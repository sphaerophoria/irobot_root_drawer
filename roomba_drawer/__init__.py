import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math

DST_SIZE = 100


def _resize_img(input_image):
    input_height, input_width, _ = input_image.shape
    resize_multplier = min(DST_SIZE / input_width, DST_SIZE / input_height)
    return cv2.resize(
        input_image,
        (int(input_width * resize_multplier), int(input_height * resize_multplier)),
    )


def _get_stroke_mask(img):
    """
    Given an input image img, generate a mask representing all strokes needed
    to reproduce the image

    :param img: Input image
    :return: A binary mask where white pixels should be drawn, and black should not
    """
    # I dunno wtf is going on https://learnopencv.com/edge-detection-using-opencv/
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    # https://stackoverflow.com/questions/56394869/how-to-estimate-piece-wise-smooth-fit-to-a-noisy-mask
    skeletonized = skeletonize(edges, method="lee")
    return skeletonized


def _get_strokes_from_mask(stroke_mask):
    """
    :param stroke_mask: Binary mask of all strokes
    :return: List of list of points. Each item in the outer list represents a
    path, each element in the inner list represents an XY position in the stroke
    """
    strokes = []

    # NOTE: using RETR_EXTERNAL will result in some types of contours not being found.
    # E.g. pupils in eyes may be missed. The alternative would be to use a more
    # comprehensive output and determine after if contours are too close together.
    # Since this is mostly done as a joke I don't want to invest effort into doing math
    contours, _ = cv2.findContours(
        stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # https://stackoverflow.com/questions/47936474/is-there-a-function-similar-to-opencv-findcontours-that-detects-curves-and-repla
        # Why am I assuming the loops aren't closed? Is there a way to tell
        contour_closed = False
        arclen = cv2.arcLength(contour, contour_closed)
        # Tuned on single image, should be input maybe
        eps = 0.003
        epsilon = arclen * eps
        approx = cv2.approxPolyDP(contour, epsilon, contour_closed)
        strokes.append(approx[:, 0, :])

    return strokes


def _convert_strokes_to_robot_space(strokes, image_size):
    """
    We define the robot space as distance from the center, however when we
    first generate the strokes they are in an image space where the top left corner
    is [0, 0]. Offset all the stroke positions so that [0, 0] is the center, and the
    top left corner is [-width / 2, -height / 2]

    :param strokes: List[List[x, y]]
    :param image_size: Tuple[w, h]
    :return: List[List[x',y']]
    """

    new_strokes = []
    image_w, image_h = image_size

    offset_x = -image_w / 2
    offset_y = -image_h / 2

    for stroke in strokes:
        new_stroke = []
        for point in stroke:
            new_stroke.append([point[0] + offset_x, point[1] + offset_y])

        new_strokes.append(np.array(new_stroke))

    return new_strokes


def _normalize_rotation(rot_rad):
    """
    Move rotation into [-pi, pi] so that we're always rotating by the minimum amount required
    """

    # Ensure we're in [ 0, 2pi ]
    while rot_rad > 2 * math.pi:
        rot_rad -= 2 * math.pi
    while rot_rad < 0:
        rot_rad += 2 * math.pi

    if rot_rad > math.pi:
        rot_rad = -(2 * math.pi - rot_rad)

    return rot_rad


def save_strokes(strokes, output_path):
    """
    Render the strokes on a graph and save the plot to output_path
    """
    plt.clf()
    for stroke in strokes:
        # [ [x,y], [x,y], [x,y] ...]
        xs = stroke[..., 0]
        ys = stroke[..., 1]
        plt.plot(xs, ys)
    plt.savefig(output_path)
    plt.clf()


def generate_strokes(img_path: str, save_intermediate_to=None):
    """
    Generate strokes that will approximate the input image

    :param img_path: Path to an input image to draw
    :return: List of list of points. Each item in the outer list represents a
    path, each element in the inner list represents an XY position in the stroke
    """

    img = cv2.imread(img_path)
    img = _resize_img(img)
    stroke_mask = _get_stroke_mask(img)
    strokes = _get_strokes_from_mask(stroke_mask)

    if save_intermediate_to is not None:
        save_intermediate_to = Path(save_intermediate_to)
        save_intermediate_to.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_intermediate_to / "resized.png"), img)
        cv2.imwrite(str(save_intermediate_to / "stroke_mask.png"), stroke_mask)
        save_strokes(strokes, save_intermediate_to / "image_space_strokes.png")

    return _convert_strokes_to_robot_space(strokes, (img.shape[1], img.shape[0]))


COMMAND_MOVE = 0
COMMAND_ROTATE = 1
COMMAND_MARKER_UP = 2
COMMAND_MARKER_DOWN = 3


class RobotMoveGenerator:
    def __init__(self):
        # Position is modeled as the center being 0,0
        self.position = np.array([0, 0])
        # Normally we would express this with vectors and linear algebra, but this is
        # doable with high school math
        self.rotation_radians = 0

    def _move_to_position(self, pos):
        moves = []
        relative_movement = np.array(pos) - self.position
        # Pythagoras
        relative_movement_distance = math.sqrt(
            relative_movement[0] ** 2 + relative_movement[1] ** 2
        )

        # Need to rotate so we are facing in the direction of the relative movement
        new_rotation_absolute = math.atan2(relative_movement[1], relative_movement[0])

        # And now calculate the rotation relative to our current rotation
        move_rot_rad = new_rotation_absolute - self.rotation_radians
        move_rot_rad = _normalize_rotation(move_rot_rad)

        # Generate the commands to issue the move
        moves.append((COMMAND_ROTATE, move_rot_rad))
        moves.append((COMMAND_MOVE, relative_movement_distance))

        # Save the current rotation/position
        self.position = pos
        self.rotation_radians = new_rotation_absolute

        return moves

    def draw_stroke(self, stroke):
        """
        Generates a list of commands needed to draw the given stroke

        :param stroke: List[Point]
        :return: List[Tuple] where each elements is of the form (COMMAND_XYZ, params...)
        """
        moves = []

        # Move to the first position in the stroke before dropping the marker
        stroke_iter = iter(stroke)
        first_point = next(stroke_iter)

        moves.extend(self._move_to_position(first_point))
        moves.append((COMMAND_MARKER_DOWN,))

        for point in stroke_iter:
            moves.extend(self._move_to_position(point))

        moves.append((COMMAND_MARKER_UP,))

        return moves


class RobotCommandRunner:
    def __init__(self, robot):
        self.robot = robot

    async def issue_command(self, command):
        if command[0] == COMMAND_MARKER_UP:
            await self.robot.marker.up()
        elif command[0] == COMMAND_MARKER_DOWN:
            await self.robot.marker.down()
        elif command[0] == COMMAND_MOVE:
            distance_mm = command[1] * 10
            # According to https://github.com/iRobotEducation/root-robot-ble-protocol
            # the movement should be in mm. Hardcode 1 pixel == 1cm for now
            await self.robot.motor.drive(distance_mm)
        elif command[0] == COMMAND_ROTATE:
            rotation_deg = command[1] * math.pi / 180.0
            rotation_decideg = rotation_deg * 10
            # According to https://github.com/iRobotEducation/root-robot-ble-protocol
            # the rotation should be in deci-degrees
            await self.robot.motor.rotate(rotation_decideg)
        else:
            raise RuntimeError(f"Unknown command: {command[0]}")
