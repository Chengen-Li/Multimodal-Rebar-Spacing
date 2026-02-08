from ultralytics import YOLO
import torch
from torchvision.ops import nms
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import circmean, circstd
import json
import yaml


def get_bounding_boxes(image, model_path, logger=None):
    """
    Args:
        image: Input image (can be file path string, numpy array, or PIL image)
        model_path: Path to the YOLO model weights file
        logger: Logger instance for logging detection information

    Returns:
        numpy.ndarray: Array of bounding boxes in xyxy format
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = YOLO(model_path).to(device)
    results = model.predict(image, verbose=False)

    boxes = results[0].boxes
    scores = boxes.conf.detach().clone()
    xyxy = boxes.xyxy.detach().clone()

    # Apply Non-Maximum Suppression (NMS)
    # Removes redundant bounding boxes that detect the same object multiple times, keeping only the best detection
    mask = nms(xyxy, scores, iou_threshold=0.3)
    filtered_boxes = xyxy[mask]

    if results[0].boxes is not None:
        result = filtered_boxes.cpu().numpy()
        if logger is not None:
            logger.info(f"intersects: {result}")
        return result
    else:
        raise ValueError("No bounding boxes detected in the image.")


def rotate(vector, angle):
    """
    Rotates a 2D vector by a specified angle.

    Args:
        vector (list): A 2D vector [x, y]
        angle (float): Rotation angle in degrees (positive = counterclockwise)

    Returns:
        list: Rotated vector [new_x, new_y]
    """
    [x, y] = vector
    angler = angle * np.pi / 180
    newx = x * np.cos(angler) - y * np.sin(angler)
    newy = x * np.sin(angler) + y * np.cos(angler)
    return [newx, newy]


def get_cone_boundaries(vector, angle):
    """
    Creates a span (cone) around a given vector by rotating it by ±angle degrees.

    This function takes a 2D vector and creates two boundary vectors by rotating
    the original vector clockwise and counterclockwise by the specified angle.
    The resulting span represents a cone or wedge shape that can be used to
    check if other vectors fall within this angular range.

    Args:
        vector (list or np.array): A 2D vector [x, y] that serves as the center direction
        angle (float): The rotation angle in degrees (±angle creates the span width)
                      For example, angle=10 creates a 20-degree cone (±10°)

    Returns:
        np.array: A 2x2 matrix where:
                 - First column: vector rotated by +angle degrees (counterclockwise)
                 - Second column: vector rotated by -angle degrees (clockwise)

    Example:
        >>> span = get_cone_boundaries([1, 0], 30)  # 30° rotation around horizontal vector
        >>> print(span)
        [[    0.86603,     0.86603],
         [        0.5,        -0.5]]
    """
    # Rotate the input vector by +angle degrees (counterclockwise)
    positive_rotation = rotate(vector, angle)
    # Rotate the input vector by -angle degrees (clockwise)
    negative_rotation = rotate(vector, -angle)

    # Stack the two rotated vectors to form a 2x2 matrix
    # This matrix defines the boundary vectors of the span/cone
    return np.column_stack((positive_rotation, negative_rotation))


def vector_aligned_with_pc(
    vertex1, vertex2, principal_component, tolerance_angle
) -> bool:
    """
    Checks if the vector between two vertices aligns with a principal component within a tolerance angle.

    Args:
        vertex1 (np.array): First vertex coordinates [x, y] (start point)
        vertex2 (np.array): Second vertex coordinates [x, y] (end point)
        principal_component (np.array): Principal component vector [x, y] from PCA
        tolerance_angle (float): Angular tolerance in degrees (±tolerance creates acceptance cone)

    Returns:
        bool: True if vector from vertex1 to vertex2 is within the acceptance cone, False otherwise

    Example:
        >>> vector_aligned_with_pc([0, 0], [0.9, 0.2], [1, 0], 30)  # Check if vector is within ±30° of x-axis
        True
        >>> vector_aligned_with_pc([0, 0], [0.2, 0.9], [1, 0], 30)  # Vector at ~77° from x-axis
        False
    """
    # Calculate the direction vector from vertex1 to vertex2
    direction_vector = np.array(vertex2) - np.array(vertex1)

    # Create the span (cone boundaries) by rotating the principal component vector ±angle degrees
    cone = get_cone_boundaries(principal_component, tolerance_angle)

    try:
        x, y = np.linalg.solve(cone, direction_vector)
        return x >= 0 and y >= 0

    except np.linalg.LinAlgError:
        return False


def norm_radian(radian, pi=np.pi):
    """
    Normalizes an angle in radians to the range [-π, π].

    This function ensures that any input angle (in radians) is wrapped into the standard interval
    from -π to π. This is useful for comparing angles and performing circular statistics,
    as angles outside this range are equivalent to some angle within it.

    Args:
        radian (float): The angle in radians to normalize.
        pi (float, optional): The value of π to use (default: math.pi).

    Returns:
        float: The normalized angle in radians, guaranteed to be within [-π, π].

    Example:
        >>> norm_radian(4)
        -2.2831853071795862
        >>> norm_radian(-4)
        2.2831853071795862
        >>> norm_radian(3.14)
        3.14
    """
    radian = radian % (2 * pi)
    return radian if abs(radian) <= pi else radian - (1 if radian >= 0 else -1) * 2 * pi


def get_circular_outlier_indices(radians, threshold: float = 1.5):
    """
    Identifies outlier indices in a list of angles using circular statistics.

    This function computes the circular mean and circular standard deviation of the input angles,
    then flags as outliers any angles whose deviation from the mean exceeds (threshold * circular std).
    Useful for filtering out lines or vectors whose orientation is inconsistent with the main group.

    Args:
        radians (list or np.ndarray): List/array of angles in radians (e.g., from np.arctan2).
        threshold (float): Multiplier for the circular standard deviation to set the outlier threshold. Default is 1.5.

    Returns:
        list: Indices of input angles that are considered outliers.

    Example:
        >>> radians = [0.1, 0.2, 0.15, 3.0]
        >>> outliers = get_circular_outlier_indices(radians, threshold=1.5)
        >>> print(outliers)
        [3]
    """
    if threshold == 0:
        return []

    radians = [2 * radian for radian in radians]
    mean = circmean(radians, high=np.pi, low=-np.pi)
    maxdelta = threshold * circstd(radians, high=np.pi, low=-np.pi)
    deltas = [norm_radian(radian - mean) for radian in radians]
    outlier_indices = [
        i for i, z in enumerate(zip(radians, deltas)) if abs(z[1]) > maxdelta
    ]

    return outlier_indices


def get_mode_outlier_indices(radians, threshold: float = 10):
    """
    Identifies outlier indices in a list of angles using histogram-based mode detection.

    This function computes a histogram of the input angles, finds the mode (most frequent bin),
    and flags as outliers any angles whose deviation from the mode exceeds the specified threshold.
    This approach is useful for filtering out lines or vectors whose orientation is inconsistent
    with the most common direction in the dataset.

    Args:
        radians (list or np.ndarray): List/array of angles in radians (e.g., from np.arctan2).
        threshold (float, optional): Angular threshold in degrees from the mode. Default is 10.

    Returns:
        list: Indices of input angles that are considered outliers.

    Example:
        >>> radians = [0.1, 0.15, 0.12, 1.5, 0.11]  # Most angles around 0.1, one at 1.5
        >>> outliers = get_mode_outlier_indices(radians, threshold=15)
        >>> print(outliers)
        [3]  # Index of the 1.5 radian angle
    """
    if threshold == 0:
        return []

    bins = 36
    # Compute histogram
    hist, bin_edges = np.histogram(radians, bins=bins, range=(-np.pi, np.pi))
    # Find the index of the bin with the most values
    max_bin_index = np.argmax(hist)
    mode = bin_edges[max_bin_index] + (np.pi / bins)  # Center of the mode bin

    threshold = threshold * np.pi / 180
    outlier_indices = [
        i
        for i, radian in enumerate(radians)
        if abs(norm_radian(radian - mode)) > threshold
    ]

    return outlier_indices


def remove_outliers(operation, points, threshold):
    """
    Removes outlier line segments based on their angular orientation using statistical methods.

    This function calculates the angle of each line segment and applies a specified outlier
    detection algorithm to identify and filter out lines whose orientations deviate
    significantly from the main directional pattern in the dataset.

    Args:
        operation (callable): Outlier detection function that takes (radians, threshold)
                             and returns indices of outlier angles.
                             Examples: get_circular_outlier_indices, get_mode_outlier_indices
        points (list): List of line segments where each segment is [[x1, y1], [x2, y2]]
        threshold (float or None): Statistical threshold for outlier detection.
                                  - For circular method: multiplier for circular std deviation
                                  - For mode method: angular threshold in degrees from mode
                                  - If None or 0, no filtering is performed

    Returns:
        list: Filtered list of line segments with outliers removed, preserving the original
              [[x1, y1], [x2, y2]] format

    Example:
        >>> lines = [[[0, 0], [10, 1]], [[0, 0], [10, 0]], [[0, 0], [1, 10]]]
        >>> filtered = remove_outliers(get_mode_outlier_indices, lines, 15)
        # Removes the nearly vertical line [[[0, 0], [1, 10]]] if it deviates
        # more than 15° from the dominant horizontal direction

    Note:
        - Line angles are computed using np.arctan2(dy, dx) from start to end point
        - Returns original list unchanged if threshold is None or 0
        - Preserves order of remaining line segments after outlier removal
    """
    if threshold:
        radians = []
        for shape in points:
            point1 = np.array(shape[0])
            point2 = np.array(shape[1])
            # Because we detect vertex using pc, so the direction will be pretty much same
            vector = point2 - point1
            radian = np.arctan2(vector[1], vector[0])
            radians.append(radian)
        # Remove outliers
        outlier_indices = operation(radians, threshold=threshold)
        points = [points[i] for i in range(len(points)) if i not in outlier_indices]

    return points


def prune_lines_using_hough_transform(image_path, points, threshold):
    """
    Filters line segments using Hough transform analysis to remove lines with inconsistent orientations.

    This function performs a two-stage filtering process:
    1. Initial statistical outlier removal using mode-based angle filtering
    2. Hough transform analysis to determine the dominant line orientations in the image
    3. Final filtering to keep only lines whose angles fall within the detected orientation range
    4. Sorts remaining lines by their distance to detected Hough lines

    The process works by:
    - Drawing the initially filtered lines on a canvas
    - Applying Hough transform to detect dominant line orientations
    - Finding angle peaks that represent the most prominent line directions
    - Filtering the original line segments to keep only those aligned with detected orientations
    - Sorting by distance to the nearest Hough line

    Args:
        image_path (str): Path to the input image file
        pc_points (list): List of line segments where each segment is [[x1, y1], [x2, y2]]
        threshold (float): Statistical threshold for initial mode-based outlier removal.
                          If 0 or None, no initial filtering is performed.

    Returns:
        list: Filtered list of line segments that align with the dominant orientations
              detected by the Hough transform, preserving the original [[x1, y1], [x2, y2]] format

    Example:
        >>> pc1_points = [[[0, 0], [10, 1]], [[5, 5], [15, 6]], [[0, 0], [1, 10]]]
        >>> filtered = prune_lines_using_hough_transform('image.jpg', pc1_points, 10)
        # Returns only lines that align with the dominant horizontal orientation
        # detected by the Hough transform

    Note:
        - Uses a 5-degree tolerance around the detected angle range for final filtering
        - Handles angle wrapping around π/-π boundary by checking both angle and angle+π
        - Canvas thickness of 3 pixels is used for reliable Hough transform detection
        - Hough peaks threshold is set to 50% of maximum accumulator value
    """
    # Perform Hough Transform pruning on pc
    pruned_points = remove_outliers(get_mode_outlier_indices, points, threshold)

    image_path = cv2.imread(image_path)
    height, width = image_path.shape[:2]
    canvas = np.zeros((height, width), dtype=np.float32)
    for bounding_box in pruned_points:
        pt1 = (int(bounding_box[0][0]), int(bounding_box[0][1]))
        pt2 = (int(bounding_box[1][0]), int(bounding_box[1][1]))

        # Draw line on canvas with thickness
        cv2.line(canvas, pt1, pt2, 255, thickness=3)

    # Get Hough Transform data
    hspace, angles, dists = hough_line(canvas)
    # Find peaks
    _, thetas, rhos = hough_line_peaks(
        hspace, angles, dists, threshold=0.5 * np.max(hspace)
    )
    # Get Hough Transform angle range
    radians = []
    for theta in thetas:
        radian = norm_radian(theta + np.pi / 2, pi=np.pi)
        radians.append(radian)
    diff = np.ptp(radians)
    if diff > np.pi / 2:
        for i in range(len(radians)):
            if radians[i] > np.pi / 2:
                radians[i] = norm_radian(radians[i] + np.pi, pi=np.pi)
    min_radian, max_radian = np.min(radians), np.max(radians)

    # Filter lines within the angle range with some tolerance
    tolerance_degrees = 5
    tolerance_rad = tolerance_degrees * np.pi / 180
    expanded_min = min_radian - tolerance_rad
    expanded_max = max_radian + tolerance_rad

    # Get Hough lines info
    hough_line_equations = []
    for rho, theta in zip(rhos, thetas):
        line = convert_rho_theta_to_line_equation(rho, theta)
        hough_line_equations.append(line)

    # Perform final filtering
    results = [[] for _ in range(len(hough_line_equations))]
    for i, line in enumerate(points):
        point1 = np.array(line[0])
        point2 = np.array(line[1])
        # Calculate the angle of the line
        vector = point2 - point1
        radian = np.arctan2(vector[1], vector[0])
        if (
            expanded_min <= radian <= expanded_max
            or expanded_min <= norm_radian(radian + np.pi, np.pi) <= expanded_max
        ):
            mid_point = (point1 + point2) / 2
            x, y = mid_point[0], mid_point[1]

            min_distance = float("inf")
            hough_line_index = -1  # Index of the closest Hough line

            for j, hough_line_equation in enumerate(hough_line_equations):
                A, B, C = hough_line_equation
                distance = abs(A * x + B * y - C) / np.sqrt(A**2 + B**2)
                if distance < min_distance:
                    min_distance = distance
                    hough_line_index = j
            if hough_line_index != -1:
                results[hough_line_index].append(line)

    return results


def convert_rho_theta_to_line_equation(rho, theta) -> tuple:
    """
    Converts a line defined by polar coordinates (rho, theta) to its linear equation
    in Cartesian coordinates (Ax + By = C).

    Args:
        rho (float): The perpendicular distance from the origin to the line.
        theta (float): The angle (in radians) of the normal vector from the origin to the line.

    Returns:
        tuple: A tuple (A, B, C) representing the line equation Ax + By = C.
    """
    A = np.cos(theta)
    B = np.sin(theta)
    C = rho
    return A, B, C


def prune_lines_by_length(pc_points, threshold=2.0):
    # Get all lengths
    lengths = []
    for point in pc_points:
        point1 = np.array(point[0])
        point2 = np.array(point[1])
        length = np.linalg.norm(point2 - point1, ord=2)
        lengths.append(length)
    lengths = np.array(lengths)

    median = np.median(lengths)
    mad = np.median(np.abs(lengths - median))

    # Define acceptable range
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad

    # Filter lines within acceptable length range
    filtered_points = []
    for i, length in enumerate(lengths):
        if lower_bound <= length <= upper_bound:
            filtered_points.append(pc_points[i])

    return filtered_points


def add_points_to_shapes(shapes, points, pc_direction, logger=None):
    """
    Adds grouped line segments to the shapes list in JSON-compatible format.

    This function takes a list of Hough transform grouped line segments and appends
    them to the provided shapes list. Each group is represented as a dictionary
    containing the grouped lines, orientation (horizontal or vertical), and shape type.
    The function handles the conversion from NumPy data types to JSON-serializable
    Python types for proper JSON encoding.

    Args:
        shapes (list): The list to which line group dictionaries will be appended.
        points (list): List of line groups, where each group contains multiple line
                      segments [[x1, y1], [x2, y2]] that share similar Hough transform
                      characteristics (proximity to the same detected Hough line).
        pc_direction (str): Principal component direction ("left", "right", "up", or "down")
                           used to determine the overall orientation of the line group.
        logger: Logger instance for logging detection information

    Returns:
        list: The updated 'shapes' list with the new line groups added.
    """
    # for line in points:
    #     shapes.append(
    #         {
    #             "points": [
    #                 [float(line[0][0]), float(line[0][1])],
    #                 [float(line[1][0]), float(line[1][1])],
    #             ],
    #             "orientation": horizontal_or_vertical(pc_direction),
    #             "shape_type": "line",
    #         }
    #     )
    for group in points:
        lines = []
        for line in group:
            lines.append(
                [
                    [float(line[0][0]), float(line[0][1])],
                    [float(line[1][0]), float(line[1][1])],
                ]
            )

        dists = []
        for line in group:
            point1 = np.array(line[0])
            point2 = np.array(line[1])
            length = np.linalg.norm(point2 - point1, ord=2)
            dists.append(length)
        
        if logger is not None:
            logger.info(f"line_pairs: {lines}")
            logger.info(f"line_dists: {dists}")

        shapes.append(
            {
                "lines": lines,
                "orientation": horizontal_or_vertical(pc_direction),
                "shape_type": "group of lines",
            }
        )

    return shapes


def get_lines(image_path, model_path, logger=None, threshold: float = 10) -> str:
    """
    Detects rebar intersections and generates grouped connection lines using PCA alignment and Hough transform analysis.

    This function performs comprehensive rebar intersection detection and line generation through
    a multi-stage process:

    1. **Object Detection**: Uses YOLO model to detect rebar intersection bounding boxes
    2. **Vertex Extraction**: Extracts center points (vertices) from detected bounding boxes
    3. **Principal Component Analysis**: Applies PCA to identify the two dominant directional patterns
    4. **Neighbor Finding**: For each vertex, finds nearest neighbors aligned with PC1 and PC2 directions
    5. **Line Generation**: Creates line segments connecting aligned vertices using bounding box edges
    6. **Hough Transform Grouping**: Groups lines by their proximity to detected Hough lines and sorts by distance

    The algorithm ensures that generated lines connect actual rebar intersections by:
    - Using 30-degree tolerance cones for directional alignment checking
    - Connecting bounding box edges rather than centers for more accurate line placement
    - Applying Hough transform analysis to group lines by their dominant orientations
    - Sorting line groups by distance to detected Hough lines (closest first)

    Args:
        image_path (str): Path to the input image file (supports common formats: jpg, png, etc.)
        model_path (str): Path to the trained YOLO model weights file (.pt format)
        logger: Logger instance for logging detection information
        threshold (float, optional): Statistical threshold for mode-based outlier detection.
                                   Higher values are more permissive. Default is 10 degrees.

    Returns:
        str: JSON string containing detected line groups in the format:
            {
                "shapes": [
                    {
                        "lines": [[[x1, y1], [x2, y2]], [[x3, y3], [x4, y4]], ...],
                        "orientation": "horizontal" | "vertical",
                        "shape_type": "group of lines"
                    },
                    ...
                ]
            }

    Example:
        >>> json_result = get_lines('image_path.jpg', 'model.pt', threshold=15)
        >>> import json
        >>> data = json.loads(json_result)
        >>> for i, shape in enumerate(data['shapes']):
        ...     print(f"Group {i}: {len(shape['lines'])} lines, {shape['orientation']}")
    """
    vertices = []
    shapes = []
    pc1_points = []
    pc2_points = []
    
    config = load_config()

    # Get bounding boxes from the image using the model
    bounding_boxes = get_bounding_boxes(image_path, model_path, logger)
    vertices = []
    for box in bounding_boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        vertices.append((x_center, y_center))
    vertices = np.array(vertices)

    # PCA Alignment
    mean = np.mean(vertices, axis=0)
    vc = vertices - mean
    _, _, Vh = np.linalg.svd(vc)  # Get the principal component vectors
    pc1, pc2 = Vh[0], Vh[1]
    # Get pc vector using "left", "right", "up" or "down"
    pc_direction = {
        "pc1": get_pc_direction(pc1),
        "pc2": get_pc_direction(pc2),
    }

    # Find the nearest vertices in the direction of each principal component for each vertex
    for i, vertex in enumerate(vertices):
        pc1_vertex = {
            "index": None,
            "distance": float("inf"),
        }
        pc2_vertex = {
            "index": None,
            "distance": float("inf"),
        }
        for j, other_vertex in enumerate(vertices):
            if i == j:
                continue
            # Search for closest vertex in pc1 direction
            if vector_aligned_with_pc(vertex, other_vertex, pc1, 30):
                diff = np.array(other_vertex) - np.array(vertex)
                distance = np.linalg.norm(diff, ord=2)
                if distance < pc1_vertex["distance"]:
                    # pc1_vertex["vertex"] = tuple(other_vertex)
                    pc1_vertex["index"] = j
                    pc1_vertex["distance"] = distance
            # Search for closest vertex in pc2 direction
            if vector_aligned_with_pc(vertex, other_vertex, pc2, 30):
                diff = np.array(other_vertex) - np.array(vertex)
                distance = np.linalg.norm(diff, ord=2)
                if distance < pc2_vertex["distance"]:
                    pc2_vertex["index"] = j
                    pc2_vertex["distance"] = distance

        # Store the point into pc1_points or pc2_points respectfully
        if pc1_vertex["index"] is not None:
            direction = pc_direction["pc1"]
            start = bounding_boxes[i]
            end = bounding_boxes[pc1_vertex["index"]]

            # # Get points from the edge of bounding boxes instead of center points
            # points = get_points_from_direction(start, end, direction)

            # Get points from center points
            start_vertice = get_vertice_from_box(start)
            end_vertice = get_vertice_from_box(end)
            points = [start_vertice, end_vertice]

            pc1_points.append(points)
        if pc2_vertex["index"] is not None:
            direction = pc_direction["pc2"]
            start = bounding_boxes[i]
            end = bounding_boxes[pc2_vertex["index"]]

            # # Get points from the edge of bounding boxes instead of center points
            # points = get_points_from_direction(start, end, direction)

            # Get points from center points
            start_vertice = get_vertice_from_box(start)
            end_vertice = get_vertice_from_box(end)
            points = [start_vertice, end_vertice]

            pc2_points.append(points)

    # Removing outliers depending on the radian of vector for pc1_points and pc2_points using get_circular_outlier_indices or get_mode_outlier_indices
    # pc1_points = remove_outliers(
    #     get_circular_outlier_indices, pc1_points, threshold
    # )
    # pc2_points = remove_outliers(
    #     get_circular_outlier_indices, pc2_points, threshold
    # )
    # pc1_points = remove_outliers(
    #     get_mode_outlier_indices, pc1_points, threshold
    # )
    # pc2_points = remove_outliers(
    #     get_mode_outlier_indices, pc2_points, threshold
    # )

    if config["features"]["prune_by_length"]:
        pc1_points = prune_lines_by_length(pc1_points)
        pc2_points = prune_lines_by_length(pc2_points)

    # Perform Hough Transform pruning on pc1_points and pc2_points
    pc1_points = prune_lines_using_hough_transform(
        image_path, pc1_points, threshold=threshold
    )
    pc2_points = prune_lines_using_hough_transform(
        image_path, pc2_points, threshold=threshold
    )

    add_points_to_shapes(shapes, pc1_points, pc_direction["pc1"], logger)
    add_points_to_shapes(shapes, pc2_points, pc_direction["pc2"], logger)

    return json.dumps({"shapes": shapes}, indent=2)


def point_on_hough_line(x, y, theta, rho, tolerance=1e-2):
    return abs(x * np.cos(theta) + y * np.sin(theta) - rho) < tolerance


def get_pc_direction(pc) -> str:
    """
    Determines the directional orientation of a principal component vector.

    This function analyzes the components of a 2D principal component vector
    to classify its direction as one of four cardinal directions: "left", "right", "up", or "down".
    The classification is based on which component (x or y) has the larger absolute value,
    and the sign of that component.

    Args:
        pc (list or np.array): A 2D principal component vector [x, y]

    Returns:
        str: Direction of the vector, one of "left", "right", "up", or "down"

    Example:
        >>> get_pc_direction([1, 0.1])
        'right'
        >>> get_pc_direction([-1, 0.1])
        'left'
        >>> get_pc_direction([0.1, 1])
        'down'
        >>> get_pc_direction([0.1, -1])
        'up'

    Note:
        - If |x| >= |y|: returns "right" (x >= 0) or "left" (x < 0)
        - If |y| > |x|: returns "down" (y >= 0) or "up" (y < 0)
    """
    x, y = pc
    if abs(x) >= abs(y):
        return "right" if x >= 0 else "left"
    else:
        return "down" if y >= 0 else "up"


def get_points_from_direction(start, end, direction):
    """
    Returns the coordinates of two points on the edges of two bounding boxes,
    allowing a line to be drawn between the edges in a specified direction.

    Args:
        start (list or np.ndarray): The first bounding box in [x1, y1, x2, y2] format.
        end (list or np.ndarray): The second bounding box in [x1, y1, x2, y2] format.
        direction (str): The direction of the connection ("left", "right", "up", or "down").

    Returns:
        list: A list of two [x, y] points, one on the edge of each bounding box.

    Example:
        >>> get_points_from_direction([10, 20, 30, 40], [50, 60, 70, 80], "right")
        [[30, 30.0], [50, 70.0]]

    Note:
        This function is useful for visualizing connections between objects by drawing lines
        that start and end at the edges of bounding boxes, rather than at their centers.
    """
    if direction == "left":
        return [[start[0], (start[1] + start[3]) / 2], [end[2], (end[1] + end[3]) / 2]]
    elif direction == "right":
        return [
            [start[2], (start[1] + start[3]) / 2],
            [end[0], (end[1] + end[3]) / 2],
        ]
    elif direction == "up":
        return [
            [(start[0] + start[2]) / 2, start[1]],
            [(end[0] + end[2]) / 2, end[3]],
        ]
    elif direction == "down":
        return [
            [(start[0] + start[2]) / 2, start[3]],
            [(end[0] + end[2]) / 2, end[1]],
        ]


def horizontal_or_vertical(direction) -> str:
    if direction == "left" or direction == "right":
        return "horizontal"
    else:
        return "vertical"


def get_vertice_from_box(box) -> list:
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return [x_center, y_center]


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config