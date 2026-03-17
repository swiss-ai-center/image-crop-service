from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import cv2
import numpy as np
import json
from common_code.tasks.service import get_extension

api_description = """
This service crops images. It takes an image and a list of int coordinates (x1, y1, x2, y2)
and returns the cropped image.
"""
api_summary = """
Crops images with the specified area.
"""

api_title = "Image Crop API."
version = "1.0.0"

settings = get_settings()


class MyService(Service):
    """
    Image crop model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Image Crop",
            slug="image-crop",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
                FieldDescription(name="area", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING
                ),
            ],
            has_ai=False,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/image-crop/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        raw = data["image"].data
        input_type = data["image"].type
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
        raw_areas = data["area"].data
        area = json.loads(raw_areas)['area']

        guessed_extension = get_extension(input_type)
        is_success, cropped_image = cv2.imencode(guessed_extension, img[area[1]:area[3], area[0]:area[2]])

        return {
            "result": TaskData(
                data=cropped_image.tobytes(),
                type=input_type,
            )
        }
