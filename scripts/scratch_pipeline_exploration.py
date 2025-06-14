import numpy as np
import cv2 # OpenCV
import fitz  # PyMuPDF

from pathlib import Path
from typing import Any

from pprint import pp

TEMPLATE_PATH = Path("Field-Datasheet-Current-ver-4.4-1-29-2025_FORM_FILLABLE.pdf")
SCANNED_PDF_PATH = Path("images/ideal_form_scanned_600_DPI_color.pdf")
TEMPLATE_DPI = 600
SCANNED_DPI = 600


def main() -> None:
    # Build template.
    template_doc = fitz.open(TEMPLATE_PATH)
    template_page = template_doc[0]

    ## Rasterize template PDF to image.
    template_gray = rasterize_PDF_to_image(page=template_page, dpi=TEMPLATE_DPI)
    window = cv2.namedWindow("Template Gray", cv2.WINDOW_NORMAL)
    cv2.imshow("Template Gray", template_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Get template fiducial points.
    template_fiducial_points = get_fiducial_points(img=template_gray)

    ## Get template ROI coordinates.
    widgets = template_page.widgets()
    ROIs = {}
    for widget in widgets:
        field_rect = convert_fitz_rect_to_pixel_rect(
            fitz_rect=widget.rect,
            fitz_page=template_page,
            image_shape=template_gray
        )
        ROIs[widget.field_name] = {"template_rect": field_rect}

    pp(ROIs)

    # Align input image to template.
    input_doc = fitz.open(SCANNED_PDF_PATH)
    input_page = input_doc[0]

    ## Rasterize scanned PDF to image.
    input_img_gray = rasterize_PDF_to_image(page=input_page, dpi=SCANNED_DPI)
    window = cv2.namedWindow("Input Image Gray", cv2.WINDOW_NORMAL)
    cv2.imshow("Input Image Gray", input_img_gray)
    cv2.waitKey(0)
    cv2.destroyWindow(window)

    ## Warp image to template.
    inpiut_img_fiducial_points = get_fiducial_points(img=input_img_gray)

    M = cv2.getPerspectiveTransform(src=inpiut_img_fiducial_points, dst=template_fiducial_points)
    input_img_warped = cv2.warpPerspective(
        src=input_img_gray,
        M=M,
        dsize=(template_gray.shape[1], template_gray.shape[0])
    )
    window = cv2.namedWindow("Input Image Warped", cv2.WINDOW_NORMAL)
    cv2.imshow("Input Image Warped", input_img_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract ROIs from aligned image.

    ## Crop to ROIs.
    for roi_field_name, template_rect in ROIs.items():
        x0, y0, x1, y1 = template_rect["template_rect"]
        roi = input_img_warped[y0:y1, x0:x1]
        ROIs[roi_field_name]["roi"] = roi

    pp(ROIs)

    ## Save ROI images.
    for roi_field_name, roi_data in ROIs.items():
        roi_img = roi_data["roi"]
        cv2.imwrite(Path(f"scratch/roi_{roi_field_name}.jpg"), roi_img)

    # Vizualize results using Ipython display.
    for roi_field_name, roi_data in ROIs.items():
        roi_img_path = Path(f"scratch/roi_{roi_field_name}.jpg")
        roi_img = cv2.imread(str(roi_img_path))
        window = cv2.namedWindow(f"ROI {roi_field_name}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"ROI {roi_field_name}", roi_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def rasterize_PDF_to_image(
    page: fitz.Page,
    dpi: int
) -> cv2.Mat | np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]:
    scale = dpi / 72  # PDF “points” are 1/72 inch
    mat  = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
    if pix.n == 4:  # Drop alpha channel if present.
        img_arr = img_arr[:, :, :3]

    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    return img_gray


def get_fiducial_points(
    img: cv2.Mat | np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Detect three fiducial points in the image."""
    _, img_thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    img_thresh = cv2.medianBlur(img_thresh, 3)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Image shape: {img.shape}")

    border_corners = []
    for contour in contours:
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            print(f"area: {area}")
            # Adjust these area thresholds to match the printed mark size at 300 DPI
            # if 500 < area < 5000:
            print(f"x: {x}, y: {y}, w: {w}, h: {h}, area: {area}")
            print (f"w/h: {w/h}")
            # if 0.8 < w/h < 1.2:
            M = cv2.moments(approx)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            breakpoint()
            border_corners.append([cx, cy])

    print(border_corners)

    if len(border_corners) != 4:
        raise RuntimeError(f"Expected 4 fiducials in the blank template, found {len(border_corners)}")

    fiducial_points = np.array([
        [0, 0],          # Top-left
        # [w - 1, 0],     # Top-right
        [w - 1, h - 1], # Bottom-right
        [0, h - 1]      # Bottom-left
    ], dtype=np.float32)

    return fiducial_points


def convert_fitz_rect_to_pixel_rect(
    fitz_rect: fitz.Rect,
    fitz_page: fitz.Page,
    image_shape: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
) -> tuple[int, int, int, int]:
    """Convert a PyMuPDF Rect to pixel coordinates in a cv2 image."""
    width_scale = image_shape.shape[1] / fitz_page.rect.width
    height_scale = image_shape.shape[0] / fitz_page.rect.height

    x0 = int(np.round(fitz_rect.x0 * width_scale))
    y0 = int(np.round(fitz_rect.y0 * height_scale))
    x1 = int(np.round(fitz_rect.x1 * width_scale))
    y1 = int(np.round(fitz_rect.y1 * height_scale))

    return (x0, y0, x1, y1)


if __name__ == "__main__":
    main()