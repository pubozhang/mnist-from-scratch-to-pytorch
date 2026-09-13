import pygame
import numpy as np
import sys
from inference import MODEL_LOADER, predict_digit

WINDOW_SIZE = 800
CANVAS_SIZE = 600
IMAGE_SIZE = 28
CANVAS_OFFSET = 100
DRAW_RADIUS = 22
DISPLAY_SCALE = CANVAS_SIZE / IMAGE_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)

CONVOLUTION_KERNEL = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.float32,
)

def conv2d_single_channel(image: np.ndarray, kernel: np.ndarray, padding: int = 1, stride: int = 1):
    """Apply one 2D kernel to one grayscale image."""
    if image.ndim != 2:
        raise ValueError(f"image must be a 2D array, but received shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be a 2D array, but received shape {kernel.shape}")

    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, pad_width=padding, mode="constant", constant_values=0, )

    output_height = (padded_image.shape[0] - kernel_height ) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    feature_map = np.zeros((output_height, output_width), dtype=np.float32,)

    for output_row in range(output_height):
        for output_column in range(output_width):

            input_row = output_row * stride
            input_column = output_column * stride

            image_region = padded_image[input_row:input_row + kernel_height, input_column:input_column + kernel_width,]

            feature_map[output_row, output_column] = np.sum(image_region * kernel)

    return feature_map

def normalize_feature_map(feature_map):
    """Scale an arbitrary feature map to the range used by the display."""
    feature_map = np.asarray(feature_map, dtype=np.float32)

    minimum = np.min(feature_map)
    maximum = np.max(feature_map)

    if maximum == minimum:
        return np.zeros_like(feature_map)

    normalized = (feature_map - minimum) / (maximum - minimum)
    return normalized

def shrink(canvas):
    """Downsample the 600 x 600 drawing canvas to a 28 x 28 image."""
    result = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    block_size = CANVAS_SIZE // IMAGE_SIZE

    for y in range(28):
        for x in range(28):
            y_start = y * block_size
            y_end = (y + 1) * block_size

            x_start = x * block_size
            x_end = (x + 1) * block_size

            block = canvas[y_start:y_end, x_start:x_end]
            result[y, x] = np.mean(block)

    return result

def center_image(image_28x28):
    """Move the image's center of mass to the center of a 28 x 28 frame."""
    total_intensity = np.sum(image_28x28)
    if total_intensity == 0:
        return image_28x28

    ny, nx = image_28x28.shape
    y_indices, x_indices = np.indices((ny, nx))

    center_x = np.sum(x_indices * image_28x28) / total_intensity
    center_y = np.sum(y_indices * image_28x28) / total_intensity

    shift_x = int(round(14 - center_x))
    shift_y = int(round(14 - center_y))

    centered_image = np.zeros_like(image_28x28)
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            new_y = y + shift_y
            new_x = x + shift_x
            if 0 <= new_y < IMAGE_SIZE and 0 <= new_x < IMAGE_SIZE:
                centered_image[new_y, new_x] = image_28x28[y, x]

    return centered_image

def main():
    model = MODEL_LOADER()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Digit Drawing Board")
    controls_font = pygame.font.Font(None, 24)

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE))
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    display_scale = CANVAS_SIZE / IMAGE_SIZE

    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r:
                    canvas = np.zeros((600, 600))
                    image = np.zeros((28, 28))

                elif event.key == pygame.K_SPACE:
                    probabilities, predicted_digit = predict_digit(image, model)
                    for digit in range(10):
                        print(f"{digit}: {probabilities[0, digit].item() * 100:.2f}%")
                    print(f"Prediction: {predicted_digit}")

                elif event.key == pygame.K_l:
                    image = normalize_feature_map(conv2d_single_channel(image, CONVOLUTION_KERNEL))

        if drawing:

            mouse_x, mouse_y = pygame.mouse.get_pos()

            mouse_x -= 100
            mouse_y -= 100

            r = 22

            for y in range(max(0, mouse_y - r), min(mouse_y + r, 600)):
                for x in range(max(0, mouse_x - r), min(mouse_x + r, 600)):

                    if (mouse_x - x) ** 2 + (mouse_y - y) ** 2 <= r ** 2:
                        canvas[y, x] = 1

            image = shrink(canvas)

            image = center_image(image)

        screen.fill(WHITE)

        for y in range(28):
            for x in range(28):
                val = int(image[y, x] * 255)

                clr = (val, val, val)

                rect_x = int(x * display_scale + 100)
                rect_y = int(y * display_scale + 100)

                rect_size = int(display_scale) + 1

                pygame.draw.rect(
                    screen,
                    clr,
                    (rect_x, rect_y, rect_size, rect_size)
                )

        pygame.draw.rect(screen, GREY, (CANVAS_OFFSET, CANVAS_OFFSET, CANVAS_SIZE, CANVAS_SIZE), 5)

        controls_surface = controls_font.render(
            "Left mouse: draw    R: reset    L: convolution    Space: predict",
            True,
            BLACK,
        )
        screen.blit(controls_surface, (CANVAS_OFFSET, 735))

        pygame.display.update()

if __name__ == "__main__":
    main()
