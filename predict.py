# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import shutil
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

    def load_image(self, path):
        print(f"load_image from {path}")
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        ref_image: Path = Input(description="Image with faces"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        face_image = self.load_image(ref_image)
        face_image = numpy.asarray(face_image)
        face_embeddings = get_face_embedding(face_image)
        images = inference(prompt, face_embeddings)
        output_paths = []
        for image in images:
            output_path = "/tmp/out-{}.png".format(uuid.uuid1())
            image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
