# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import numpy
import uuid
from typing import List

from cog import BasePredictor, Input, Path

from diffusers.utils import load_image

from inference import get_face_embedding, inference


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        ref_image: Path = Input(description="Image with faces"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        face_image = load_image(ref_image)
        face_image = numpy.asarray(face_image)
        face_embeddings = get_face_embedding(face_image)
        images = inference(prompt, face_embeddings)
        output_paths = []
        for image in images:
            output_path = "/tmp/out-{}.png".format(uuid.uuid1())
            image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
